package com.example.dps_zn

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: GraphicOverlayView
    private lateinit var settingsButton: ImageButton
    private lateinit var ttsIndicator: ImageButton
    private lateinit var prefs: AppPreferences
    private lateinit var announcer: SpeechAnnouncer

    private var analyzer: ImageAnalyzer? = null

    // Прединфлейченный диалог настроек: создаётся один раз и переиспользуется,
    // чтобы не дёргать UI-поток на inflate/measure при каждом открытии.
    private var settingsDialog: AlertDialog? = null
    private var settingsView: View? = null
    private var dlgTtsSwitch: SwitchCompat? = null
    private var dlgTtsStatus: TextView? = null
    private var dlgTtsTestBtn: Button? = null
    private var dlgTtsInstallBtn: Button? = null
    private var dlgConfSeek: SeekBar? = null
    private var dlgConfValue: TextView? = null

    private val labelLastSeenAtMs = HashMap<String, Long>()
    private val labelLastSpokenAtMs = HashMap<String, Long>()

    /** Только кадры ImageAnalysis — не смешивать с блокирующим getInstance(). */
    private val analysisExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "dps-image-analysis").apply { isDaemon = true }
    }

    /** Один постоянный listener, апдейтит и значок в шапке, и текст в открытом диалоге. */
    private val ttsStatusListener: (SpeechAnnouncer.Status) -> Unit = { st ->
        runOnUiThread {
            updateTtsIndicator()
            if (settingsDialog?.isShowing == true) applyDialogTtsStatus(st)
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) startCamera()
        else Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        prefs = AppPreferences(this)
        announcer = SpeechAnnouncer(this).apply { setEnabled(prefs.ttsEnabled) }

        previewView = findViewById(R.id.preview_view)
        overlay = findViewById(R.id.graphic_overlay)
        settingsButton = findViewById(R.id.settings_button)
        ttsIndicator = findViewById(R.id.tts_indicator)
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        settingsButton.setOnClickListener { showSettingsDialog() }
        ttsIndicator.setOnClickListener { toggleTtsQuick() }

        announcer.setStatusListener(ttsStatusListener)
        updateTtsIndicator()

        // Лениво готовим диалог после первого кадра — без блока главного потока на старте,
        // а к моменту первого тапа по шестерёнке inflate уже сделан.
        previewView.post { preInflateSettingsDialog() }

        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED -> startCamera()
            else -> permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onStop() {
        super.onStop()
        announcer.stopSpeaking()
    }

    override fun onDestroy() {
        super.onDestroy()
        settingsDialog?.takeIf { it.isShowing }?.dismiss()
        announcer.shutdown()
        analysisExecutor.shutdown()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            {
                try {
                    val provider = future.get()
                    bindCameraUseCases(provider)
                } catch (e: Exception) {
                    Toast.makeText(this, "Камера: ${e.message}", Toast.LENGTH_LONG).show()
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun bindCameraUseCases(provider: ProcessCameraProvider) {
        val targetRotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
        val preview = Preview.Builder()
            .setTargetRotation(targetRotation)
            .build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

        val analysis = ImageAnalysis.Builder()
            .setTargetRotation(targetRotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val newAnalyzer = ImageAnalyzer(applicationContext) { iw, ih, dets ->
            overlay.post {
                overlay.setSourceImageSize(iw, ih)
                overlay.setDetections(dets)
            }
            handleDetectionsForSpeech(dets)
        }
        newAnalyzer.confidenceThreshold = prefs.confidenceThreshold
        analysis.setAnalyzer(analysisExecutor, newAnalyzer)
        analyzer = newAnalyzer

        provider.unbindAll()
        if (tryBind(provider, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)) {
            overlay.setMirrorX(false)
            return
        }

        provider.unbindAll()
        if (tryBind(provider, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)) {
            overlay.setMirrorX(true)
            return
        }

        Toast.makeText(
            this,
            "Камера (задняя/фронтальная): не удалось привязать use cases",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun tryBind(
        provider: ProcessCameraProvider,
        selector: CameraSelector,
        preview: Preview,
        analysis: ImageAnalysis
    ): Boolean {
        return try {
            provider.bindToLifecycle(this, selector, preview, analysis)
            true
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Решаем, что озвучивать: только знаки, у которых в последние [NEW_APPEARANCE_GAP_MS]
     * не было видимости (т.е. знак «появился заново»). Дополнительно — кулдаун [SPEAK_COOLDOWN_MS]
     * внутри MainActivity и аналогичный кулдаун внутри [SpeechAnnouncer]. Двойная защита: если
     * один из тиков просочится из-за дрожания трекинга, второй гарантированно отсечёт повтор.
     */
    private fun handleDetectionsForSpeech(dets: List<GraphicOverlayView.OverlayDetection>) {
        if (!announcer.isEnabled() || !announcer.isReady()) return
        val now = SystemClock.elapsedRealtime()

        val current = HashSet<String>(dets.size * 2)
        for (d in dets) current.add(d.label)

        for (label in current) {
            val lastSeen = labelLastSeenAtMs[label] ?: 0L
            val lastSpoken = labelLastSpokenAtMs[label] ?: 0L
            val isNewAppearance = (now - lastSeen) > NEW_APPEARANCE_GAP_MS
            val canSpeak = (now - lastSpoken) > SPEAK_COOLDOWN_MS
            if (isNewAppearance && canSpeak) {
                if (announcer.announce(label)) {
                    labelLastSpokenAtMs[label] = now
                }
            }
            labelLastSeenAtMs[label] = now
        }

        if (labelLastSeenAtMs.size > MAX_TRACKED_LABELS) {
            val expiry = now - LABEL_EXPIRY_MS
            labelLastSeenAtMs.entries.removeAll { it.value < expiry }
            labelLastSpokenAtMs.entries.removeAll { it.value < expiry }
        }
    }

    private fun toggleTtsQuick() {
        val newValue = !announcer.isEnabled()
        prefs.ttsEnabled = newValue
        announcer.setEnabled(newValue)
        updateTtsIndicator()
        if (newValue && !announcer.isReady()) {
            Toast.makeText(this, R.string.tts_test_no_engine, Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateTtsIndicator() {
        val on = announcer.isEnabled() && announcer.isReady()
        ttsIndicator.setImageResource(if (on) R.drawable.ic_volume_up else R.drawable.ic_volume_off)
    }

    /**
     * Готовим View и AlertDialog один раз, привязываем все listener'ы. При каждом
     * открытии остаётся только обновить состояние из prefs и показать.
     */
    private fun preInflateSettingsDialog() {
        if (settingsDialog != null) return

        val view = LayoutInflater.from(this).inflate(R.layout.dialog_settings, null, false)
        val ttsSwitch = view.findViewById<SwitchCompat>(R.id.tts_switch)
        val ttsStatus = view.findViewById<TextView>(R.id.tts_status)
        val ttsTestBtn = view.findViewById<Button>(R.id.tts_test_button)
        val ttsInstallBtn = view.findViewById<Button>(R.id.tts_install_button)
        val confSeek = view.findViewById<SeekBar>(R.id.conf_seekbar)
        val confValue = view.findViewById<TextView>(R.id.conf_value)

        confSeek.max = MAX_CONF_PERCENT - MIN_CONF_PERCENT
        confSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                confValue.text = formatPercent(MIN_CONF_PERCENT + progress)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        ttsTestBtn.setOnClickListener {
            if (!announcer.speakTest()) {
                Toast.makeText(this, R.string.tts_test_no_engine, Toast.LENGTH_SHORT).show()
            }
        }
        ttsInstallBtn.setOnClickListener { openTtsSettings() }

        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.settings_title)
            .setView(view)
            .setPositiveButton(R.string.settings_save) { _, _ -> commitSettingsFromDialog() }
            .setNegativeButton(R.string.settings_cancel, null)
            .create()

        settingsView = view
        settingsDialog = dialog
        dlgTtsSwitch = ttsSwitch
        dlgTtsStatus = ttsStatus
        dlgTtsTestBtn = ttsTestBtn
        dlgTtsInstallBtn = ttsInstallBtn
        dlgConfSeek = confSeek
        dlgConfValue = confValue
    }

    private fun showSettingsDialog() {
        if (settingsDialog == null) preInflateSettingsDialog()
        val dialog = settingsDialog ?: return
        val ttsSwitch = dlgTtsSwitch ?: return
        val confSeek = dlgConfSeek ?: return
        val confValue = dlgConfValue ?: return

        ttsSwitch.isChecked = prefs.ttsEnabled
        val pct = (prefs.confidenceThreshold * 100f).toInt()
            .coerceIn(MIN_CONF_PERCENT, MAX_CONF_PERCENT)
        confSeek.progress = pct - MIN_CONF_PERCENT
        confValue.text = formatPercent(pct)
        applyDialogTtsStatus(announcer.status)

        dialog.show()
    }

    private fun applyDialogTtsStatus(s: SpeechAnnouncer.Status) {
        val statusText = dlgTtsStatus ?: return
        val testBtn = dlgTtsTestBtn ?: return
        val (textRes, colorInt, canTest) = when (s) {
            SpeechAnnouncer.Status.INITIALIZING ->
                Triple(R.string.settings_tts_status_initializing, COLOR_NEUTRAL, false)
            SpeechAnnouncer.Status.READY_RU ->
                Triple(R.string.settings_tts_status_ready_ru, COLOR_OK, true)
            SpeechAnnouncer.Status.READY_FALLBACK ->
                Triple(R.string.settings_tts_status_fallback, COLOR_WARN, true)
            SpeechAnnouncer.Status.UNAVAILABLE ->
                Triple(R.string.settings_tts_status_unavailable, COLOR_ERR, false)
        }
        statusText.setText(textRes)
        statusText.setTextColor(colorInt)
        testBtn.isEnabled = canTest
    }

    private fun commitSettingsFromDialog() {
        val ttsSwitch = dlgTtsSwitch ?: return
        val confSeek = dlgConfSeek ?: return

        val tts = ttsSwitch.isChecked
        val pct = MIN_CONF_PERCENT + confSeek.progress
        val confF = pct / 100f
        prefs.ttsEnabled = tts
        prefs.confidenceThreshold = confF
        announcer.setEnabled(tts)
        analyzer?.confidenceThreshold = confF
        updateTtsIndicator()
    }

    /**
     * Пытаемся открыть наиболее подходящую системную страницу: если русский голос «не загружен»
     * (READY_FALLBACK) — сразу установщик данных движка. Иначе — общую страницу TTS-настроек.
     * Если оба интента не приняты ни одной activity на устройстве — показываем Toast.
     */
    private fun openTtsSettings() {
        val candidates = mutableListOf<Intent>()
        if (announcer.status == SpeechAnnouncer.Status.READY_FALLBACK) {
            candidates += Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA)
        }
        candidates += Intent("com.android.settings.TTS_SETTINGS")
        candidates += Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)
        for (i in candidates) {
            try {
                i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(i)
                return
            } catch (_: Throwable) { /* пробуем следующий */ }
        }
        Toast.makeText(this, R.string.tts_open_failed, Toast.LENGTH_SHORT).show()
    }

    private fun formatPercent(p: Int): String =
        String.format(Locale.getDefault(), "%d%%", p)

    companion object {
        private const val NEW_APPEARANCE_GAP_MS = 1_500L
        private const val SPEAK_COOLDOWN_MS = 6_000L
        private const val MAX_TRACKED_LABELS = 64
        private const val LABEL_EXPIRY_MS = 30_000L

        private const val MIN_CONF_PERCENT = 30
        private const val MAX_CONF_PERCENT = 85

        private const val COLOR_OK = 0xFF2E7D32.toInt()       // зелёный — всё в порядке
        private const val COLOR_WARN = 0xFFB26A00.toInt()     // янтарный — фолбэк
        private const val COLOR_ERR = 0xFFC62828.toInt()      // красный — движок не работает
        private const val COLOR_NEUTRAL = Color.DKGRAY
    }
}
