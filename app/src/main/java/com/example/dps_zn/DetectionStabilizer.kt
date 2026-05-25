package com.example.dps_zn

import android.graphics.RectF

/**
 * Стабилизация выхода YOLO между кадрами, чтобы рамки не «мигали» 8–10 раз в секунду.
 *
 * Идея — простой трекинг с гистерезисом:
 *  - Свежий бокс не показываем сразу: только после [minHitsToShow] кадров подряд,
 *    в которых модель его подтвердила (то же имя класса + IoU >= [matchIoU]).
 *  - Потерянный бокс не убираем сразу: даём ему «погаснуть» в течение
 *    [maxMissesToHide] кадров без совпадения. Это убирает мерцание на дрожащих кадрах.
 *  - Координаты бокса и confidence сглаживаем по EMA с коэффициентом [smoothingAlpha],
 *    чтобы рамка не дёргалась от кадра к кадру.
 *
 * Не thread-safe: используется только из analysis-потока [ImageAnalyzer].
 */
class DetectionStabilizer(
    private val minHitsToShow: Int = 2,
    private val maxMissesToHide: Int = 4,
    private val matchIoU: Float = 0.3f,
    private val smoothingAlpha: Float = 0.5f
) {

    private data class Track(
        val label: String,
        val box: RectF,
        var confidence: Float,
        var hits: Int = 1,
        var misses: Int = 0,
        var visible: Boolean = false
    )

    private val tracks = ArrayList<Track>()

    /**
     * Подаёт детекции одного кадра, возвращает то, что должно быть отрисовано.
     */
    fun update(detections: List<GraphicOverlayView.OverlayDetection>):
        List<GraphicOverlayView.OverlayDetection> {

        // Сначала «штрафуем» все треки за пропуск; если матч найдётся ниже — обнулим miss.
        for (t in tracks) t.misses += 1

        val matched = HashSet<Int>()
        for (det in detections) {
            // Ищем самый похожий ещё не занятый трек с тем же лейблом.
            var bestIdx = -1
            var bestIoU = 0f
            for (i in tracks.indices) {
                if (i in matched) continue
                val t = tracks[i]
                if (t.label != det.label) continue
                val iou = computeIoU(t.box, det.rectImage)
                if (iou >= matchIoU && iou > bestIoU) {
                    bestIoU = iou
                    bestIdx = i
                }
            }
            if (bestIdx >= 0) {
                val t = tracks[bestIdx]
                emaInPlace(t.box, det.rectImage, smoothingAlpha)
                t.confidence = smoothingAlpha * t.confidence + (1f - smoothingAlpha) * det.confidence
                t.hits += 1
                t.misses = 0
                if (!t.visible && t.hits >= minHitsToShow) t.visible = true
                matched.add(bestIdx)
            } else {
                tracks.add(
                    Track(
                        label = det.label,
                        box = RectF(det.rectImage),
                        confidence = det.confidence,
                        hits = 1
                    )
                )
            }
        }

        // Удаляем треки, которые слишком долго не подтверждались.
        var i = 0
        while (i < tracks.size) {
            if (tracks[i].misses > maxMissesToHide) tracks.removeAt(i) else i++
        }

        // Наружу — только подтверждённые. Копируем RectF, чтобы внешний код не мутировал состояние.
        val out = ArrayList<GraphicOverlayView.OverlayDetection>(tracks.size)
        for (t in tracks) {
            if (t.visible) {
                out.add(GraphicOverlayView.OverlayDetection(t.label, t.confidence, RectF(t.box)))
            }
        }
        return out
    }

    fun reset() {
        tracks.clear()
    }

    private fun emaInPlace(prev: RectF, next: RectF, alpha: Float) {
        val a = alpha
        val b = 1f - alpha
        prev.set(
            a * prev.left + b * next.left,
            a * prev.top + b * next.top,
            a * prev.right + b * next.right,
            a * prev.bottom + b * next.bottom
        )
    }

    private fun computeIoU(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }
}
