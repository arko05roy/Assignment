package com.example.facereel.processing

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.min

object FaceQuality {
    fun score(
        crop: Bitmap,
        frameWidth: Int,
        frameHeight: Int,
        yaw: Float?,
        roll: Float?,
    ): Pair<Double, Double> {
        val area = (crop.width.toDouble() * crop.height / (frameWidth * frameHeight).toDouble())
            .coerceIn(0.0, 1.0)
        val sharpness = laplacianVariance(crop)
        return sharpness to qualityRank(area, sharpness, yaw, roll)
    }

    private fun laplacianVariance(bitmap: Bitmap): Double {
        val width = min(bitmap.width, 96)
        val height = min(bitmap.height, 96)
        if (width < 3 || height < 3) return 0.0
        val scaled = if (bitmap.width == width && bitmap.height == height) bitmap
        else Bitmap.createScaledBitmap(bitmap, width, height, true)
        val values = DoubleArray((width - 2) * (height - 2))
        var index = 0
        for (y in 1 until height - 1) for (x in 1 until width - 1) {
            val c = luminance(scaled.getPixel(x, y))
            val laplacian = luminance(scaled.getPixel(x - 1, y)) + luminance(scaled.getPixel(x + 1, y)) +
                luminance(scaled.getPixel(x, y - 1)) + luminance(scaled.getPixel(x, y + 1)) - 4 * c
            values[index++] = laplacian.toDouble()
        }
        if (scaled !== bitmap) scaled.recycle()
        val mean = values.average()
        return values.sumOf { (it - mean) * (it - mean) } / values.size
    }

    private fun luminance(color: Int): Int =
        ((color shr 16 and 0xff) * 299 + (color shr 8 and 0xff) * 587 + (color and 0xff) * 114) / 1000
}

/** Sharpness dominates representative selection; size is only a tie-breaker. */
internal fun qualityRank(area: Double, sharpness: Double, yaw: Float?, roll: Float?): Double {
    val normalizedSharpness = (sharpness / 150.0).coerceIn(0.0, 1.0)
    val posePenalty = ((abs(yaw ?: 0f) + abs(roll ?: 0f)) / 90.0).coerceIn(0.0, 1.0)
    val frontal = 1.0 - posePenalty
    return 0.15 * area + 0.65 * normalizedSharpness + 0.20 * frontal
}
