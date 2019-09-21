package com.kokleong.mtskotlinarcore.tensorflow.lite

import android.graphics.Bitmap

interface IClassifier {
    data class Recognition(
        var id: String ="",
        var title : String = "",
        var confidence: Float = 0f,
        var quant :Boolean = false
    ){
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence)";
        }

    }
    fun recognizeImage(bitmap : Bitmap) : List<Recognition>

    fun close()
}