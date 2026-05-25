package com.example.dps_zn

import android.content.Context

/**
 * Тонкая обёртка над SharedPreferences для пользовательских настроек.
 * Хранит флаги озвучки и порог уверенности детекции.
 */
class AppPreferences(context: Context) {

    private val prefs = context.applicationContext
        .getSharedPreferences(NAME, Context.MODE_PRIVATE)

    var ttsEnabled: Boolean
        get() = prefs.getBoolean(KEY_TTS_ENABLED, DEFAULT_TTS_ENABLED)
        set(value) {
            prefs.edit().putBoolean(KEY_TTS_ENABLED, value).apply()
        }

    var confidenceThreshold: Float
        get() = prefs.getFloat(KEY_CONF_THRESHOLD, DEFAULT_CONF_THRESHOLD)
            .coerceIn(MIN_CONF_THRESHOLD, MAX_CONF_THRESHOLD)
        set(value) {
            val clamped = value.coerceIn(MIN_CONF_THRESHOLD, MAX_CONF_THRESHOLD)
            prefs.edit().putFloat(KEY_CONF_THRESHOLD, clamped).apply()
        }

    companion object {
        private const val NAME = "dps_zn_prefs"
        private const val KEY_TTS_ENABLED = "tts_enabled"
        private const val KEY_CONF_THRESHOLD = "conf_threshold"

        const val DEFAULT_TTS_ENABLED = true
        const val DEFAULT_CONF_THRESHOLD = 0.50f
        const val MIN_CONF_THRESHOLD = 0.30f
        const val MAX_CONF_THRESHOLD = 0.85f
    }
}
