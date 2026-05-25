package com.example.dps_zn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Анализатор кадра камеры: YOLOv8 TFLite (INT8/FLOAT32) → letterbox → инференс → NMS → overlay.
 *
 * Особенности:
 *  - Модель и метки загружаются лениво при первом кадре.
 *  - Универсальный парсер выхода: [1, 4+nc, A] и [1, A, 4+nc], FLOAT32/UINT8/INT8.
 *  - Имена классов берутся из labels.txt (латинские), для отрисовки переводятся в русские
 *    через [RUSSIAN_LABELS]. Если перевода нет — показываем исходное имя.
 */
class ImageAnalyzer(
    context: Context,
    private val onResults: (imageWidth: Int, imageHeight: Int, detections: List<GraphicOverlayView.OverlayDetection>) -> Unit
) : ImageAnalysis.Analyzer {

    private val appContext = context.applicationContext

    private val initLock = Any()
    private var interpreter: Interpreter? = null
    private var classNames: List<String> = emptyList()

    private var inputDataType: DataType = DataType.FLOAT32
    private var inputQuantScale: Float = 1f
    private var inputQuantZeroPoint: Int = 0
    private var inputWidth: Int = MODEL_SIZE
    private var inputHeight: Int = MODEL_SIZE

    private var outputDataType: DataType = DataType.FLOAT32
    private var outputQuantScale: Float = 1f
    private var outputQuantZeroPoint: Int = 0
    private var outputShape: IntArray = intArrayOf()
    private var outputTotal: Int = 0
    private var outputClassCount: Int = 0

    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var outputFloats: FloatArray = FloatArray(0)
    private var pixelBuffer: IntArray = IntArray(0)

    private val lastTelemetryAtMs = ConcurrentHashMap<String, Long>()
    private val telemetryCooldownMs = 10_000L
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /**
     * Сглаживает выход YOLO между кадрами: рамка появляется только после нескольких
     * подтверждений и держится ещё несколько кадров после потери. Используется только
     * из analysis-потока, поэтому без синхронизации.
     */
    private val stabilizer = DetectionStabilizer()

    /** Порог уверенности можно менять на лету из настроек; читается на каждом кадре. */
    @Volatile
    var confidenceThreshold: Float = DEFAULT_CONF_THRESHOLD

    @Volatile
    private var loggedMeta = false
    private var frameIndex = 0L

    private fun ensureEngine() {
        if (interpreter != null) return
        synchronized(initLock) {
            if (interpreter != null) return

            val model = FileUtil.loadMappedFile(appContext, MODEL_FILE)
            val interp = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })

            val rawLabels = appContext.assets.open(LABELS_FILE).use { stream ->
                BufferedReader(InputStreamReader(stream)).lineSequence()
                    .map { it.trim().trimEnd('\\') }
                    .filter { it.isNotEmpty() }
                    .toList()
            }

            val inTensor = interp.getInputTensor(0)
            val outTensor = interp.getOutputTensor(0)
            val inShape = inTensor.shape()
            val outShape = outTensor.shape()

            if (inShape.size == 4) {
                inputHeight = inShape[1]
                inputWidth = inShape[2]
            }

            inputDataType = inTensor.dataType()
            outputDataType = outTensor.dataType()
            inTensor.quantizationParams().let {
                inputQuantScale = it.scale
                inputQuantZeroPoint = it.zeroPoint
            }
            outTensor.quantizationParams().let {
                outputQuantScale = it.scale
                outputQuantZeroPoint = it.zeroPoint
            }

            outputShape = outShape
            outputTotal = outShape.fold(1, Int::times)
            outputClassCount = inferClassCount(outShape)

            // Метки берём ровно столько, сколько классов у модели; лишнее в labels.txt игнорируем.
            classNames = when {
                outputClassCount > 0 && rawLabels.size >= outputClassCount -> rawLabels.take(outputClassCount)
                rawLabels.isNotEmpty() -> rawLabels
                else -> (0 until outputClassCount).map { "class_$it" }
            }

            inputBuffer = ByteBuffer.allocateDirect(inTensor.numBytes()).order(ByteOrder.nativeOrder())
            outputBuffer = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
            outputFloats = FloatArray(outputTotal)
            pixelBuffer = IntArray(inputWidth * inputHeight)

            Log.i(
                TAG,
                "model: in=${inShape.contentToString()} dtype=$inputDataType " +
                    "out=${outShape.contentToString()} dtype=$outputDataType " +
                    "classes=$outputClassCount labels.txt=${rawLabels.size}"
            )
            if (outputClassCount > 0 && rawLabels.size != outputClassCount) {
                Log.w(
                    TAG,
                    "labels.txt содержит ${rawLabels.size} строк, у модели $outputClassCount классов. " +
                        "Проверь соответствие порядка."
                )
            }

            interpreter = interp
        }
    }

    override fun analyze(imageProxy: ImageProxy) {
        try {
            ensureEngine()
            frameIndex += 1L
            if (frameIndex % INFERENCE_EVERY_N_FRAMES != 0L) return

            val interp = interpreter ?: return
            val inBuf = inputBuffer ?: return
            val outBuf = outputBuffer ?: return

            val rotation = imageProxy.imageInfo.rotationDegrees
            val rawBitmap = imageProxyToBitmap(imageProxy) ?: return
            val oriented = rotateIfNeeded(rawBitmap, rotation)
            val sourceW = oriented.width
            val sourceH = oriented.height

            val letterbox = letterboxBitmap(oriented)
            fillInputBuffer(letterbox.bitmap, inBuf)

            inBuf.rewind()
            outBuf.rewind()
            interp.run(inBuf, outBuf)
            outBuf.rewind()
            dequantizeOutput(outBuf, outputFloats)

            val boxes = parseYoloV8(outputFloats, outputShape)
            val results = ArrayList<GraphicOverlayView.OverlayDetection>(boxes.size)
            for (b in boxes) {
                val raw = classNames.getOrNull(b.classIndex) ?: continue
                if (raw.isBlank()) continue
                val display = RUSSIAN_LABELS[raw] ?: raw
                val rect = modelRectToImageRect(b.rect, letterbox)
                if (rect.width() <= 1f || rect.height() <= 1f) continue
                results.add(GraphicOverlayView.OverlayDetection(display, b.score, rect))
                maybeSendTelemetry(raw, b.score)
            }

            if (!loggedMeta) {
                loggedMeta = true
                Log.i(TAG, "first inference: boxes=${results.size} oriented=${sourceW}x${sourceH}")
            }
            val stabilized = stabilizer.update(results)
            onResults(sourceW, sourceH, stabilized)
        } catch (t: Throwable) {
            Log.e(TAG, "analyze failed", t)
        } finally {
            imageProxy.close()
        }
    }

    // -------------------------------------------------------------------------
    //  YOLOv8 head parsing
    // -------------------------------------------------------------------------

    private data class Box640(val rect: RectF, val classIndex: Int, val score: Float)

    private fun parseYoloV8(flat: FloatArray, shape: IntArray): List<Box640> {
        if (shape.size != 3 || shape[0] != 1) return emptyList()
        if (outputClassCount <= 0) return emptyList()

        val featureRows = YOLO_CLASS_START_ROW + outputClassCount
        val d1 = shape[1]
        val d2 = shape[2]

        return when {
            // [1, F, A]: индексация index = row * A + anchor
            d1 == featureRows -> parseLayout(flat, anchors = d2, rowMajor = true)
            // [1, A, F]: индексация index = anchor * F + row
            d2 == featureRows -> parseLayout(flat, anchors = d1, rowMajor = false)
            else -> {
                Log.w(TAG, "Неожиданная форма выхода: ${shape.contentToString()}, ожидалось F=$featureRows")
                emptyList()
            }
        }
    }

    private fun parseLayout(flat: FloatArray, anchors: Int, rowMajor: Boolean): List<Box640> {
        val featureRows = YOLO_CLASS_START_ROW + outputClassCount
        val classCount = outputClassCount
        val raw = ArrayList<Triple<RectF, Int, Float>>()
        val confTh = confidenceThreshold

        for (anchor in 0 until anchors) {
            var bestCls = -1
            var bestScore = 0f
            for (c in 0 until classCount) {
                val row = YOLO_CLASS_START_ROW + c
                val idx = if (rowMajor) row * anchors + anchor else anchor * featureRows + row
                if (idx >= flat.size) continue
                val s = toClassProb(flat[idx])
                if (s > bestScore) {
                    bestScore = s
                    bestCls = c
                }
            }
            if (bestCls < 0 || bestScore < confTh) continue

            val xIdx = if (rowMajor) anchor else anchor * featureRows
            val yIdx = if (rowMajor) anchors + anchor else anchor * featureRows + 1
            val wIdx = if (rowMajor) 2 * anchors + anchor else anchor * featureRows + 2
            val hIdx = if (rowMajor) 3 * anchors + anchor else anchor * featureRows + 3
            if (hIdx >= flat.size) continue

            val rect = decodeBox(flat[xIdx], flat[yIdx], flat[wIdx], flat[hIdx]) ?: continue
            raw.add(Triple(rect, bestCls, bestScore))
        }

        return nms(raw)
    }

    private fun nms(boxes: List<Triple<RectF, Int, Float>>): List<Box640> {
        if (boxes.isEmpty()) return emptyList()
        val sorted = boxes.sortedByDescending { it.third }
        val kept = ArrayList<Box640>()
        for (t in sorted) {
            if (kept.size >= MAX_DETECTIONS) break
            val r = t.first
            if (r.width() < MIN_BOX_SIZE_PX || r.height() < MIN_BOX_SIZE_PX) continue
            val area = r.width() * r.height()
            if (area > inputWidth.toFloat() * inputHeight.toFloat() * MAX_BOX_AREA_RATIO) continue
            val aspect = r.width() / r.height()
            if (aspect < MIN_BOX_ASPECT_RATIO || aspect > MAX_BOX_ASPECT_RATIO) continue

            var keep = true
            for (k in kept) {
                if (iou(r, k.rect) > NMS_IOU_THRESHOLD) {
                    keep = false
                    break
                }
            }
            if (keep) kept.add(Box640(RectF(r), t.second, t.third))
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val iw = (interRight - interLeft).coerceAtLeast(0f)
        val ih = (interBottom - interTop).coerceAtLeast(0f)
        val inter = iw * ih
        val u = a.width() * a.height() + b.width() * b.height() - inter
        return if (u <= 0f) 0f else inter / u
    }

    private fun decodeBox(p0: Float, p1: Float, p2: Float, p3: Float): RectF? {
        if (!p0.isFinite() || !p1.isFinite() || !p2.isFinite() || !p3.isFinite()) return null
        var cx = p0
        var cy = p1
        var w = p2
        var h = p3
        // Некоторые экспорты отдают координаты в [0,1].
        if (max(max(cx, cy), max(w, h)) <= 2f) {
            cx *= inputWidth
            cy *= inputHeight
            w *= inputWidth
            h *= inputHeight
        }
        if (w <= 0f || h <= 0f) return null
        val left = cx - w / 2f
        val top = cy - h / 2f
        val right = cx + w / 2f
        val bottom = cy + h / 2f
        if (right <= 0f || bottom <= 0f || left >= inputWidth || top >= inputHeight) return null
        return RectF(
            left.coerceIn(0f, inputWidth.toFloat()),
            top.coerceIn(0f, inputHeight.toFloat()),
            right.coerceIn(0f, inputWidth.toFloat()),
            bottom.coerceIn(0f, inputHeight.toFloat())
        )
    }

    private fun toClassProb(v: Float): Float {
        if (!v.isFinite()) return 0f
        // YOLOv8 экспортирует уже сигмоидальные классы. Logits страхуем через sigmoid.
        return if (v in 0f..1f) v else sigmoid(v)
    }

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    private fun inferClassCount(shape: IntArray): Int {
        if (shape.size != 3 || shape[0] != 1) return 0
        val d1 = shape[1]
        val d2 = shape[2]
        // Признаков всегда меньше, чем якорей; идентифицируем меньшее измерение.
        val featureDim = min(d1, d2)
        return (featureDim - YOLO_CLASS_START_ROW).coerceAtLeast(0)
    }

    // -------------------------------------------------------------------------
    //  Input / output buffer fill
    // -------------------------------------------------------------------------

    private fun fillInputBuffer(bitmap: Bitmap, buf: ByteBuffer) {
        val w = inputWidth
        val h = inputHeight
        val src = if (bitmap.width == w && bitmap.height == h) bitmap
                  else Bitmap.createScaledBitmap(bitmap, w, h, true)

        if (pixelBuffer.size != w * h) pixelBuffer = IntArray(w * h)
        src.getPixels(pixelBuffer, 0, w, 0, 0, w, h)

        buf.rewind()
        when (inputDataType) {
            DataType.FLOAT32 -> {
                val fb = buf.asFloatBuffer()
                for (p in pixelBuffer) {
                    fb.put(((p shr 16) and 0xFF) / 255f)
                    fb.put(((p shr 8) and 0xFF) / 255f)
                    fb.put((p and 0xFF) / 255f)
                }
            }
            DataType.UINT8 -> {
                for (p in pixelBuffer) {
                    buf.put(((p shr 16) and 0xFF).toByte())
                    buf.put(((p shr 8) and 0xFF).toByte())
                    buf.put((p and 0xFF).toByte())
                }
            }
            DataType.INT8 -> {
                // YOLOv8 INT8: scale ≈ 1/255, zeroPoint = -128 ⇒ INT8 = UINT8 − 128.
                for (p in pixelBuffer) {
                    buf.put((((p shr 16) and 0xFF) - 128).toByte())
                    buf.put((((p shr 8) and 0xFF) - 128).toByte())
                    buf.put(((p and 0xFF) - 128).toByte())
                }
            }
            else -> throw IllegalStateException("Неподдерживаемый тип входа: $inputDataType")
        }
        buf.rewind()
    }

    private fun dequantizeOutput(buf: ByteBuffer, out: FloatArray) {
        buf.rewind()
        when (outputDataType) {
            DataType.FLOAT32 -> {
                val n = min(out.size, buf.remaining() / 4)
                buf.asFloatBuffer().get(out, 0, n)
            }
            DataType.UINT8 -> {
                val n = min(out.size, buf.remaining())
                for (i in 0 until n) {
                    val raw = buf.get(i).toInt() and 0xFF
                    out[i] = (raw - outputQuantZeroPoint) * outputQuantScale
                }
            }
            DataType.INT8 -> {
                val n = min(out.size, buf.remaining())
                for (i in 0 until n) {
                    val raw = buf.get(i).toInt()
                    out[i] = (raw - outputQuantZeroPoint) * outputQuantScale
                }
            }
            else -> Log.w(TAG, "Неподдерживаемый тип выхода: $outputDataType")
        }
    }

    // -------------------------------------------------------------------------
    //  Bitmap helpers
    // -------------------------------------------------------------------------

    private data class LetterboxFrame(
        val bitmap: Bitmap,
        val scale: Float,
        val dx: Float,
        val dy: Float,
        val sourceWidth: Int,
        val sourceHeight: Int
    )

    private fun letterboxBitmap(src: Bitmap): LetterboxFrame {
        val scale = min(inputWidth / src.width.toFloat(), inputHeight / src.height.toFloat())
        val scaledW = src.width * scale
        val scaledH = src.height * scale
        val dx = (inputWidth - scaledW) / 2f
        val dy = (inputHeight - scaledH) / 2f
        val out = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(src, null, RectF(dx, dy, dx + scaledW, dy + scaledH), letterboxPaint)
        return LetterboxFrame(out, scale, dx, dy, src.width, src.height)
    }

    private fun modelRectToImageRect(rect: RectF, frame: LetterboxFrame): RectF {
        val left = ((rect.left - frame.dx) / frame.scale).coerceIn(0f, frame.sourceWidth.toFloat())
        val top = ((rect.top - frame.dy) / frame.scale).coerceIn(0f, frame.sourceHeight.toFloat())
        val right = ((rect.right - frame.dx) / frame.scale).coerceIn(0f, frame.sourceWidth.toFloat())
        val bottom = ((rect.bottom - frame.dy) / frame.scale).coerceIn(0f, frame.sourceHeight.toFloat())
        return RectF(left, top, right, bottom)
    }

    private fun rotateIfNeeded(src: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return src
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return if (image.planes.size == 1) rgbaImageProxyToBitmap(image)
               else yuv420888ToBitmap(image)
    }

    private fun rgbaImageProxyToBitmap(image: ImageProxy): Bitmap? {
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        buffer.rewind()
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val width = image.width
        val height = image.height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        if (pixelStride == 4 && rowStride == width * 4) {
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }
        val rowPixels = IntArray(width)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = buffer.get().toInt() and 0xFF
                val g = buffer.get().toInt() and 0xFF
                val b = buffer.get().toInt() and 0xFF
                val a = buffer.get().toInt() and 0xFF
                rowPixels[x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
            val rowPadding = rowStride - width * pixelStride
            if (rowPadding > 0) buffer.position(buffer.position() + rowPadding)
            bitmap.setPixels(rowPixels, 0, width, 0, y, width, 1)
        }
        return bitmap
    }

    private fun yuv420888ToBitmap(image: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(image) ?: return null
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray? {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4
        val out = ByteArray(ySize + uvSize * 2)
        val yPlane = image.planes[0]
        copyPlaneRowWise(yPlane, width, height, out, 0, 1)
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        uBuffer.rewind()
        vBuffer.rewind()
        var offset = ySize
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vu = row * vPlane.rowStride + col * vPlane.pixelStride
                val uu = row * uPlane.rowStride + col * uPlane.pixelStride
                if (vu >= vBuffer.limit() || uu >= uBuffer.limit()) return null
                out[offset++] = vBuffer.get(vu)
                out[offset++] = uBuffer.get(uu)
            }
        }
        return out
    }

    private fun copyPlaneRowWise(
        plane: ImageProxy.PlaneProxy,
        width: Int,
        height: Int,
        out: ByteArray,
        offset: Int,
        outPixelStride: Int
    ) {
        val buffer = plane.buffer.duplicate()
        buffer.rewind()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        var outIndex = offset
        for (row in 0 until height) {
            for (col in 0 until width) {
                out[outIndex] = buffer.get(row * rowStride + col * pixelStride)
                outIndex += outPixelStride
            }
        }
    }

    // -------------------------------------------------------------------------
    //  Telemetry
    // -------------------------------------------------------------------------

    private fun maybeSendTelemetry(label: String, conf: Float) {
        if (!ENABLE_TELEMETRY) return
        try {
            val now = SystemClock.elapsedRealtime()
            val prev = lastTelemetryAtMs[label] ?: 0L
            if (now - prev < telemetryCooldownMs) return
            lastTelemetryAtMs[label] = now
            FirebaseTelemetryManager.sendSignData(label, conf.toDouble())
        } catch (e: Exception) {
            Log.w(TAG, "telemetry: $e")
        }
    }

    companion object {
        private const val TAG = "ImageAnalyzer"

        private const val MODEL_FILE = "best_int8.tflite"
        private const val LABELS_FILE = "labels.txt"

        private const val MODEL_SIZE = 640
        private const val YOLO_CLASS_START_ROW = 4

        // Пороги детекции
        const val DEFAULT_CONF_THRESHOLD = 0.50f
        private const val NMS_IOU_THRESHOLD = 0.45f
        private const val MAX_DETECTIONS = 20

        // Санити-фильтры для бокса (в координатах входа модели 640×640).
        private const val MIN_BOX_SIZE_PX = 12f
        private const val MAX_BOX_AREA_RATIO = 0.85f
        private const val MIN_BOX_ASPECT_RATIO = 0.40f
        private const val MAX_BOX_ASPECT_RATIO = 2.50f

        private const val INFERENCE_EVERY_N_FRAMES = 1L
        private const val ENABLE_TELEMETRY = false

        /** Соответствие имени класса из labels.txt → подпись для отрисовки. */
        private val RUSSIAN_LABELS: Map<String, String> = mapOf(
            "asosiy_yol" to "Главная дорога",
            "avtobus_bekati" to "Остановка автобуса",
            "aylanma_harakat" to "Круговое движение",
            "aylanma_harakat_kesishuv" to "Пересечение с круговым движением",
            "bolalar" to "Дети",
            "boshqa_xavf_xatar" to "Прочие опасности",
            "chapdan_chetlab_otish" to "Объезд препятствия слева",
            "chapdan_yol_torayishi" to "Сужение дороги слева",
            "chapga_taqiqlanadi" to "Поворот налево запрещён",
            "chapga_yoki_onga" to "Движение налево или направо",
            "chorraha" to "Пересечение равнозначных дорог",
            "faqat_togri" to "Движение прямо",
            "harakatlanish_chapga" to "Движение налево",
            "harakatlanish_onga" to "Движение направо",
            "harakatlanish_taqiqlanadi" to "Движение запрещено",
            "keskin_chapga" to "Опасный поворот налево",
            "keskin_onga" to "Опасный поворот направо",
            "kirish_taqiqlanadi" to "Въезд запрещён",
            "notekis_yol" to "Неровная дорога",
            "oldinda_chapga_burilish" to "Направление поворота налево",
            "oldinda_onga_burilish" to "Направление поворота направо",
            "ondan_yoki_chapdan_chetlab_otish" to "Объезд препятствия справа или слева",
            "onga_taqiqlanadi" to "Поворот направо запрещён",
            "ongdan_chetlab_otish" to "Объезд препятствия справа",
            "ongdan_yol_torayishi" to "Сужение дороги справа",
            "ortga_qaytish" to "Разворот",
            "piyodalar_otish_joyi" to "Пешеходный переход",
            "piyodalar_otish_ogohlantirish" to "Предупреждение: пешеходный переход",
            "qayrilish_taqiqlanadi" to "Разворот запрещён",
            "qizil_chiroq" to "Красный сигнал светофора",
            "quvib_otish_taqiqlanadi" to "Обгон запрещён",
            "sariq_chiroq" to "Жёлтый сигнал светофора",
            "sirpanchiq_yol" to "Скользкая дорога",
            "stop_belgisi" to "Движение без остановки запрещено",
            "suniy_notekislik" to "Искусственная неровность",
            "svetafor_ogohlantirish" to "Светофорное регулирование",
            "tamirlash_ishlari" to "Дорожные работы",
            "tezlik_10" to "Ограничение скорости 10",
            "tezlik_20" to "Ограничение скорости 20",
            "tezlik_30" to "Ограничение скорости 30",
            "tezlik_50" to "Ограничение скорости 50",
            "tezlik_60" to "Ограничение скорости 60",
            "tezlik_70" to "Ограничение скорости 70",
            "tezlik_80" to "Ограничение скорости 80",
            "togri_yoki_chapga" to "Движение прямо или налево",
            "togri_yoki_onga" to "Движение прямо или направо",
            "toxtab_turish_taqiqlanadi" to "Стоянка запрещена",
            "toxtash_taqiqlanadi" to "Остановка запрещена",
            "turargoh" to "Парковка",
            "yashil_chiroq" to "Зелёный сигнал светофора",
            "yol_bering" to "Уступите дорогу",
            "yol_torayishi" to "Сужение дороги",
            "yovvoyi_hayvonlar_xavfi" to "Дикие животные",
            "yuk_avto_taqiqlanadi" to "Движение грузовых автомобилей запрещено",
            "zebra_chizigi" to "Зебра (разметка)"
        )
    }
}
