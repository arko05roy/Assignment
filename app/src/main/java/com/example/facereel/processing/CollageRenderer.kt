package com.example.facereel.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.File
import java.io.FileOutputStream
import kotlin.math.ceil
import kotlin.math.sqrt

object CollageRenderer {
    fun render(representatives: List<FaceCandidate>, output: File): File {
        require(representatives.isNotEmpty())
        val columns = ceil(sqrt(representatives.size.toDouble())).toInt()
        val rows = ceil(representatives.size.toDouble() / columns).toInt()
        val tile = 320
        val gutter = 12
        val bitmap = Bitmap.createBitmap(
            columns * tile + (columns + 1) * gutter,
            rows * tile + (rows + 1) * gutter,
            Bitmap.Config.ARGB_8888,
        )
        val canvas = Canvas(bitmap).apply { drawColor(Color.rgb(18, 18, 18)) }
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        representatives.forEachIndexed { index, candidate ->
            val left = gutter + (index % columns) * (tile + gutter)
            val top = gutter + (index / columns) * (tile + gutter)
            val scaled = Bitmap.createScaledBitmap(candidate.bitmap, tile, tile, true)
            canvas.drawBitmap(scaled, left.toFloat(), top.toFloat(), paint)
            if (scaled !== candidate.bitmap) scaled.recycle()
        }
        output.parentFile?.mkdirs()
        FileOutputStream(output).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        bitmap.recycle()
        return output
    }
}
