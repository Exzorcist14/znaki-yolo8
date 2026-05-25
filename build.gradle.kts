// Top-level build file. Плагины модулей определяются ниже, общие classpath-зависимости —
// в buildscript-блоке (так google-services резолвится по координатам артефакта,
// без plugin marker'а, что важно при работе через зеркала Maven Central / Google).
buildscript {
    dependencies {
        classpath("com.google.gms:google-services:4.4.2")
    }
}
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}
