package com.example.facereel.processing

import android.graphics.Bitmap
import android.graphics.Rect

data class FaceCandidate(
    val bitmap: Bitmap,
    val embedding: FloatArray,
    val timestampUs: Long,
    val trackingId: Int?,
    val bounds: Rect,
    val cropWidth: Int,
    val cropHeight: Int,
    val yawDegrees: Float?,
    val rollDegrees: Float?,
    val sharpness: Double,
    val qualityScore: Double,
)

data class FaceCluster(
    val members: MutableList<FaceCandidate>,
    var centroid: FloatArray,
) {
    val representative: FaceCandidate get() = members.maxBy { it.qualityScore }
}
