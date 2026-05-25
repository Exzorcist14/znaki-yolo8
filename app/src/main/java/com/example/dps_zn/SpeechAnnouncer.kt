package com.example.dps_zn

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Озвучка распознанных знаков через системный TextToSpeech (оффлайн).
 *
 * Защита от спама:
 *  - per-label cooldown ([cooldownMs]) — не повторять один и тот же знак чаще,
 *    чем раз в N секунд.
 *  - кап на длину очереди ([MAX_QUEUE]) — лучше услышать актуальный знак,
 *    чем дочитать тот, что уже проехал.
 *
 * Состояние движка наружу выставляется через [status]. Решение «озвучивать ли»
 * принимает MainActivity по признаку «новое появление» знака.
 */
class SpeechAnnouncer(context: Context) {

    /** Состояние TTS-движка. Наружу — только для отображения в UI. */
    enum class Status { INITIALIZING, READY_RU, READY_FALLBACK, UNAVAILABLE }

    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = null

    /** Пробовали ли уже фолбэк на дефолтный движок — чтобы не зацикливаться. */
    private var fallbackAttempted = false

    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile private var enabled: Boolean = AppPreferences.DEFAULT_TTS_ENABLED
    @Volatile var status: Status = Status.INITIALIZING
        private set

    private val pending = AtomicInteger(0)
    private val lastSpokenAt = ConcurrentHashMap<String, Long>()

    @Volatile private var statusCallback: ((Status) -> Unit)? = null

    @Volatile var cooldownMs: Long = DEFAULT_COOLDOWN_MS

    private val progressListener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {}
        override fun onDone(utteranceId: String?) {
            pending.updateAndGet { (it - 1).coerceAtLeast(0) }
        }
        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String?) {
            pending.updateAndGet { (it - 1).coerceAtLeast(0) }
        }
        override fun onError(utteranceId: String?, errorCode: Int) {
            pending.updateAndGet { (it - 1).coerceAtLeast(0) }
        }
    }

    init {
        // Шаг 1 — пробуем явно Google TTS. На «чистом» Android и большинстве вендоров он
        // совпадает с дефолтным; на китайских прошивках (Vivo/Funtouch, OPPO/ColorOS,
        // Honor и т.п.) дефолтный движок — собственный, не умеющий русского, и приложение
        // молчит, хотя в системных настройках «по умолчанию» уже выбран Google. Принудительно
        // указываем пакет, и только если его на устройстве нет — фолбэк на дефолтный.
        initWithEngine(GOOGLE_TTS_PACKAGE)
    }

    private fun initWithEngine(enginePackage: String?) {
        // Новая попытка — старый экземпляр глушим, чтобы не висели два движка.
        tts?.runCatching { shutdown() }
        tts = TextToSpeech(
            appContext,
            { initStatus -> onInitFinished(initStatus, requestedEngine = enginePackage) },
            enginePackage
        )
    }

    private fun onInitFinished(initStatus: Int, requestedEngine: String?) {
        val engine = tts ?: run {
            status = Status.UNAVAILABLE
            notifyStatus(status)
            return
        }
        if (initStatus != TextToSpeech.SUCCESS) {
            Log.w(TAG, "TTS init failed (engine=$requestedEngine, status=$initStatus)")
            // Если не получилось взять Google TTS (его нет на устройстве) — пробуем дефолтный.
            if (!fallbackAttempted && requestedEngine != null) {
                fallbackAttempted = true
                Log.i(TAG, "Fallback to default engine")
                initWithEngine(null)
                return
            }
            status = Status.UNAVAILABLE
            notifyStatus(status)
            return
        }

        engine.setSpeechRate(SPEECH_RATE)
        engine.setOnUtteranceProgressListener(progressListener)
        val newStatus = resolveLanguage(engine)
        status = newStatus
        val defaultEngineName: String? = try { engine.defaultEngine } catch (_: Throwable) { null }
        Log.i(
            TAG,
            "SpeechAnnouncer ready: status=$newStatus requested=$requestedEngine " +
                "systemDefault=$defaultEngineName"
        )
        notifyStatus(newStatus)
    }

    /**
     * Пытаемся выставить русский язык; если данных или поддержки нет — фолбэк на английский.
     * Если и английский не подгружается, движок считается непригодным.
     */
    private fun resolveLanguage(engine: TextToSpeech): Status {
        val ru = Locale.forLanguageTag("ru-RU")
        val rRu = try { engine.setLanguage(ru) } catch (t: Throwable) { TextToSpeech.LANG_NOT_SUPPORTED }
        if (rRu == TextToSpeech.LANG_AVAILABLE
            || rRu == TextToSpeech.LANG_COUNTRY_AVAILABLE
            || rRu == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
        ) {
            Log.i(TAG, "RU TTS available (code=$rRu)")
            return Status.READY_RU
        }
        Log.w(TAG, "RU TTS not available (code=$rRu) — пробуем фолбэк на английский")
        val rEn = try { engine.setLanguage(Locale.US) } catch (t: Throwable) { TextToSpeech.LANG_NOT_SUPPORTED }
        if (rEn == TextToSpeech.LANG_AVAILABLE
            || rEn == TextToSpeech.LANG_COUNTRY_AVAILABLE
            || rEn == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
        ) {
            Log.w(TAG, "EN fallback OK (code=$rEn)")
            return Status.READY_FALLBACK
        }
        Log.w(TAG, "EN fallback тоже недоступен (code=$rEn) — TTS-движок непригоден")
        return Status.UNAVAILABLE
    }

    fun setStatusListener(listener: ((Status) -> Unit)?) {
        statusCallback = listener
        listener?.invoke(status)
    }

    private fun notifyStatus(s: Status) {
        val cb = statusCallback ?: return
        mainHandler.post { cb(s) }
    }

    fun setEnabled(value: Boolean) {
        enabled = value
        if (!value) stopSpeaking()
    }

    fun isEnabled(): Boolean = enabled

    fun isReady(): Boolean = status == Status.READY_RU || status == Status.READY_FALLBACK

    /**
     * Запросить озвучку фразы. Возвращает true, если действительно поставили в очередь.
     * Cooldown по тексту защищает от повторов одного и того же знака.
     */
    fun announce(text: String): Boolean {
        if (!enabled || !isReady() || text.isBlank()) return false
        val now = SystemClock.elapsedRealtime()
        val prev = lastSpokenAt[text] ?: 0L
        if (now - prev < cooldownMs) return false
        lastSpokenAt[text] = now

        val mode = if (pending.get() >= MAX_QUEUE) {
            // Очередь забита устаревшими фразами — выбрасываем и оставляем только новую.
            pending.set(0)
            TextToSpeech.QUEUE_FLUSH
        } else {
            TextToSpeech.QUEUE_ADD
        }
        pending.incrementAndGet()
        return tryEnqueue(text, mode)
    }

    /**
     * Озвучить тестовую фразу с очисткой очереди. Используется в настройках,
     * чтобы пользователь мог сразу проверить, что движок реально говорит.
     */
    fun speakTest(): Boolean {
        if (!isReady()) return false
        val text = if (status == Status.READY_RU) TEST_PHRASE_RU else TEST_PHRASE_EN
        pending.set(1)
        return tryEnqueue(text, TextToSpeech.QUEUE_FLUSH, "test_${System.nanoTime()}")
    }

    private fun tryEnqueue(text: String, mode: Int, utteranceId: String? = null): Boolean {
        val engine = tts ?: return false
        val id = utteranceId ?: "u_${SystemClock.elapsedRealtime()}_${text.hashCode()}"
        return try {
            engine.speak(text, mode, null, id)
            true
        } catch (t: Throwable) {
            Log.w(TAG, "tts.speak failed: $t")
            pending.updateAndGet { (it - 1).coerceAtLeast(0) }
            false
        }
    }

    fun stopSpeaking() {
        try { tts?.stop() } catch (_: Throwable) { }
        pending.set(0)
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Throwable) { }
        tts = null
        pending.set(0)
        statusCallback = null
    }

    companion object {
        private const val TAG = "SpeechAnnouncer"
        private const val MAX_QUEUE = 3
        private const val SPEECH_RATE = 1.15f
        private const val TEST_PHRASE_RU = "Распознавание включено"
        private const val TEST_PHRASE_EN = "Russian voice not installed"
        const val DEFAULT_COOLDOWN_MS = 6_000L

        /**
         * Пакет Google TTS. Принудительно выбираем именно его, потому что на ряде
         * китайских прошивок (Vivo, OPPO, Honor) дефолтный системный движок не умеет
         * русский, и без явного указания приложение получает его и молчит, даже если
         * пользователь руками выбрал Google в системных настройках TTS.
         */
        private const val GOOGLE_TTS_PACKAGE = "com.google.android.tts"
    }
}
