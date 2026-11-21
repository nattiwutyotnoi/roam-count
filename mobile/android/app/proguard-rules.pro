# TFLite / GPU delegate: keep what the runtime reflects on.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options$GpuBackend
