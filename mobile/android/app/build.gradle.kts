// EXAMPLE ONLY / NOT COMPILED. App-module Gradle (Kotlin DSL). Versions are recent-ish
// defaults -- bump to whatever Android Studio's assistant recommends when you open it.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.roamcount"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.roamcount"
        minSdk = 26          // NNAPI/GPU delegate friendly
        targetSdk = 34
        versionCode = 1
        versionName = "1.0-mobile"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // Keep the model uncompressed so the Interpreter can memory-map it from assets.
    androidResources { noCompress += "tflite" }
}

dependencies {
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // TFLite / LiteRT runtime + GPU delegate.
    // Alternative (matches ultralytics' current naming): com.google.ai.edge.litert:litert
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")

    testImplementation("junit:junit:4.13.2") // for UniqueCounterTest (JVM unit test)
}
