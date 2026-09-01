package com.example.facereel.processing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.os.Build
import android.util.Log
import androidx.core.content.FileProvider
import com.example.facereel.BuildConfig
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.min

class FacePipeline(
    private val context: Context,
    private val embeddings: EmbeddingEngine,
    private val maxDurationMs: Long = MAX_DURATION_MS,
) {
    sealed interface Outcome {
        data class Success(
            val collage: File,
            val outputUri: String,
            val uniqueFaceCount: Int,
            val candidateCount: Int,
        ) : Outcome
        data object ModelUnavailable : Outcome
        data object NoFacesFound : Outcome
        data class Failure(val error: Throwable) : Outcome
    }

    suspend fun process(clip: File, onProgress: (Int) -> Unit): Outcome = withContext(Dispatchers.Default) {
        if (!embeddings.isReady) return@withContext Outcome.ModelUnavailable
        runCatching {
            val retriever = MediaMetadataRetriever()
            val detector = FaceDetection.getClient(
                FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .setMinFaceSize(0.015f)
                    .build(),
            )
            try {
                retriever.setDataSource(clip.absolutePath)
                val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull()?.let { min(it, maxDurationMs) } ?: 0L
                require(durationMs > 0) { "Recorded video has no readable duration." }
                val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                    ?.toIntOrNull() ?: 0
                val timestamps = sampleTimestamps(durationMs)
                val candidates = mutableListOf<FaceCandidate>()
                if (BuildConfig.DEBUG) {
                    File(context.cacheDir, "debug-face-v4").apply { mkdirs() }
                        .listFiles()?.forEach { it.delete() }
                }
                timestamps.forEachIndexed { index, timestampUs ->
                    val raw = frameAt(retriever, timestampUs)
                    if (raw == null) {
                        onProgress(((index + 1) * 100 / timestamps.size).coerceAtMost(100))
                        return@forEachIndexed
                    }
                    val frame = raw.rotated(rotation)
                    if (frame !== raw) raw.recycle()
                    try {
                        val faces = detector.process(InputImage.fromBitmap(frame, 0)).await()
                        val largestFaceWidth = faces.maxOfOrNull { it.boundingBox.width() } ?: 0
                        faces.filter { face ->
                            shouldKeepDetectedFace(face.boundingBox.width(), largestFaceWidth, frame.width)
                        }.sortedByDescending { it.boundingBox.width() * it.boundingBox.height() }
                            .forEachIndexed { faceRank, face ->
                            if (BuildConfig.DEBUG) {
                                Log.i(
                                    DIAGNOSTIC_TAG,
                                    "[DEBUG-face-v4] detected t=$timestampUs rank=$faceRank box=${face.boundingBox}",
                                )
                            }
                            val cropBounds = paddedAndClamped(face.boundingBox, frame.width, frame.height)
                                ?: return@forEachIndexed
                            if (cropBounds.width() < MIN_CROP_SIZE || cropBounds.height() < MIN_CROP_SIZE) return@forEachIndexed
                            val crop = Bitmap.createBitmap(frame, cropBounds.left, cropBounds.top, cropBounds.width(), cropBounds.height())
                            val cropWidth = crop.width
                            val cropHeight = crop.height
                            val (sharpness, score) = FaceQuality.score(
                                crop, frame.width, frame.height, face.headEulerAngleY, face.headEulerAngleZ,
                            )
                            val firstEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
                            val secondEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
                            val faceDown = face.getLandmark(FaceLandmark.MOUTH_BOTTOM)?.position
                                ?: face.getLandmark(FaceLandmark.NOSE_BASE)?.position
                            var displayCrop = alignedFaceCrop(
                                frame,
                                firstEye,
                                secondEye,
                                faceDown,
                                ALIGNED_DISPLAY_SIZE,
                            )
                            if (displayCrop == null) {
                                displayCrop = Bitmap.createScaledBitmap(
                                    crop, ALIGNED_DISPLAY_SIZE, ALIGNED_DISPLAY_SIZE, true,
                                ).let { scaled ->
                                    if (scaled !== crop) scaled else crop.copy(Bitmap.Config.ARGB_8888, false)
                                }
                            }
                            crop.recycle()
                            val embeddingInput = Bitmap.createScaledBitmap(
                                displayCrop, EMBEDDING_INPUT_SIZE, EMBEDDING_INPUT_SIZE, true,
                            )
                            val embedding = try {
                                if (BuildConfig.DEBUG) {
                                    FileOutputStream(
                                        File(context.cacheDir, "debug-face-v4/embedding-input-${candidates.size}.jpg"),
                                    ).use { output -> embeddingInput.compress(Bitmap.CompressFormat.JPEG, 96, output) }
                                }
                                embeddings.embed(embeddingInput)
                            } finally {
                                embeddingInput.recycle()
                            }
                            candidates += FaceCandidate(
                                bitmap = displayCrop,
                                embedding = embedding,
                                timestampUs = timestampUs,
                                trackingId = face.trackingId,
                                bounds = cropBounds,
                                cropWidth = cropWidth,
                                cropHeight = cropHeight,
                                yawDegrees = face.headEulerAngleY,
                                rollDegrees = face.headEulerAngleZ,
                                sharpness = sharpness,
                                qualityScore = score,
                            )
                        }
                    } finally {
                        frame.recycle()
                        onProgress(((index + 1) * 100 / timestamps.size).coerceAtMost(100))
                    }
                }
                if (candidates.isEmpty()) return@runCatching Outcome.NoFacesFound
                if (BuildConfig.DEBUG) {
                    logRecognitionDiagnostics(candidates)
                    writeRecognitionDiagnostics(candidates)
                }
                val clusters = FaceClusterer().cluster(candidates)
                if (BuildConfig.DEBUG) {
                    Log.i(DIAGNOSTIC_TAG, "[DEBUG-face-v4] candidates=${candidates.size} clusters=${clusters.size} sizes=${clusters.map { it.members.size }}")
                    clusters.forEachIndexed { clusterIndex, cluster ->
                        Log.i(
                            DIAGNOSTIC_TAG,
                            "[DEBUG-face-v4] cluster=$clusterIndex times=${cluster.members.map { it.timestampUs }}",
                        )
                    }
                }
                val output = File(context.cacheDir, "collages/face-reel-${System.currentTimeMillis()}.jpg")
                CollageRenderer.render(clusters.map { it.representative }, output)
                candidates.forEach { it.bitmap.recycle() }
                Outcome.Success(
                    collage = output,
                    outputUri = FileProvider.getUriForFile(context, "${context.packageName}.files", output).toString(),
                    uniqueFaceCount = clusters.size,
                    candidateCount = candidates.size,
                )
            } finally {
                detector.close()
                retriever.release()
            }
        }.getOrElse { Outcome.Failure(it) }
    }

    private fun frameAt(retriever: MediaMetadataRetriever, timestampUs: Long): Bitmap? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            retriever.getScaledFrameAtTime(
                timestampUs,
                MediaMetadataRetriever.OPTION_CLOSEST,
                SAMPLE_WIDTH,
                SAMPLE_HEIGHT,
            )
        } else {
            retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST)?.let {
                Bitmap.createScaledBitmap(it, SAMPLE_WIDTH, SAMPLE_HEIGHT, true).also { scaled -> it.recycle() }
            }
        }

    private fun Bitmap.rotated(degrees: Int): Bitmap =
        if (degrees % 360 == 0) this else Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply {
            postRotate(degrees.toFloat())
        }, true)

    /** Corrects missing/unreliable MP4 rotation metadata for the visible collage crop. */
    private fun uprightDisplayCrop(
        crop: Bitmap,
        firstEye: PointF?,
        secondEye: PointF?,
        faceDown: PointF?,
    ): Bitmap {
        if (firstEye == null || secondEye == null) return crop
        val (leftEye, rightEye) = orderedEyes(firstEye, secondEye, faceDown)
        val angle = Math.toDegrees(
            atan2(
                (rightEye.y - leftEye.y).toDouble(),
                (rightEye.x - leftEye.x).toDouble(),
            ),
        )
        if (abs(angle) < 2.0) return crop
        return Bitmap.createBitmap(crop, 0, 0, crop.width, crop.height, Matrix().apply {
            postRotate(-angle.toFloat(), crop.width / 2f, crop.height / 2f)
        }, true)
    }

    /** Creates a canonical eye-aligned face at the requested resolution. */
    private fun alignedFaceCrop(
        frame: Bitmap,
        firstEye: PointF?,
        secondEye: PointF?,
        faceDown: PointF?,
        outputSize: Int,
    ): Bitmap? {
        if (firstEye == null || secondEye == null) return null
        val (leftEye, rightEye) = orderedEyes(firstEye, secondEye, faceDown)
        val source = floatArrayOf(leftEye.x, leftEye.y, rightEye.x, rightEye.y)
        val scale = outputSize / EMBEDDING_INPUT_SIZE.toFloat()
        val target = floatArrayOf(
            EMBEDDING_LEFT_EYE_X * scale, EMBEDDING_EYE_Y * scale,
            EMBEDDING_RIGHT_EYE_X * scale, EMBEDDING_EYE_Y * scale,
        )
        val transform = Matrix().apply { setPolyToPoly(source, 0, target, 0, 2) }
        return Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888).also { output ->
            android.graphics.Canvas(output).drawBitmap(
                frame,
                transform,
                android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG),
            )
        }
    }

    private fun isUsableModelFace(bounds: Rect): Boolean =
        bounds.width() >= 24 && bounds.height() >= 24

    private fun orderedEyes(firstEye: PointF, secondEye: PointF, faceDown: PointF?): Pair<PointF, PointF> {
        if (faceDown == null) {
            return if (firstEye.x <= secondEye.x) firstEye to secondEye else secondEye to firstEye
        }
        return if (shouldKeepEyeOrder(
                firstEye.x, firstEye.y, secondEye.x, secondEye.y, faceDown.x, faceDown.y,
            )
        ) firstEye to secondEye else secondEye to firstEye
    }

    private fun logRecognitionDiagnostics(candidates: List<FaceCandidate>) {
        candidates.forEachIndexed { index, candidate ->
            val nearest = candidates.indices
                .filter { it != index }
                .maxByOrNull { cosineSimilarity(candidate.embedding, candidates[it].embedding) }
            val similarity = nearest?.let { cosineSimilarity(candidate.embedding, candidates[it].embedding) }
            Log.i(
                DIAGNOSTIC_TAG,
                "[DEBUG-face-v4] candidate=$index t=${candidate.timestampUs} track=${candidate.trackingId} sharp=${"%.1f".format(candidate.sharpness)} nearest=$nearest sim=${similarity?.let { "%.3f".format(it) }}",
            )
        }
    }

    private fun writeRecognitionDiagnostics(candidates: List<FaceCandidate>) {
        val directory = File(context.cacheDir, "debug-face-v4").apply { mkdirs() }
        File(directory, "embeddings.csv").bufferedWriter().use { writer ->
            candidates.forEachIndexed { index, candidate ->
                writer.append(index.toString()).append(',')
                    .append(candidate.timestampUs.toString()).append(',')
                    .append((candidate.trackingId ?: -1).toString()).append(',')
                    .append(candidate.sharpness.toString())
                candidate.embedding.forEach { writer.append(',').append(it.toString()) }
                writer.newLine()
                FileOutputStream(File(directory, "candidate-$index.jpg")).use { output ->
                    candidate.bitmap.compress(Bitmap.CompressFormat.JPEG, 94, output)
                }
            }
        }
    }

    private fun paddedAndClamped(bounds: Rect, width: Int, height: Int): Rect? {
        val horizontalPad = (bounds.width() * 0.20f).toInt()
        val verticalPad = (bounds.height() * 0.20f).toInt()
        val expanded = Rect(
            (bounds.left - horizontalPad).coerceAtLeast(0),
            (bounds.top - verticalPad).coerceAtLeast(0),
            (bounds.right + horizontalPad).coerceAtMost(width),
            (bounds.bottom + verticalPad).coerceAtMost(height),
        )
        return expanded.takeIf { it.width() > 0 && it.height() > 0 }
    }

    companion object {
        const val MAX_DURATION_MS = 20_000L
        private const val SAMPLE_WIDTH = 960
        private const val SAMPLE_HEIGHT = 720
        private const val MIN_CROP_SIZE = 24
        private const val EMBEDDING_INPUT_SIZE = 160
        private const val ALIGNED_DISPLAY_SIZE = 256
        private const val EMBEDDING_LEFT_EYE_X = 48.57f
        private const val EMBEDDING_RIGHT_EYE_X = 111.43f
        private const val EMBEDDING_EYE_Y = 61.43f
        private const val DIAGNOSTIC_TAG = "FaceRecognition"
    }
}

internal fun sampleTimestamps(durationMs: Long): List<Long> =
    generateSequence(0L) { it + 250_000L }
        .takeWhile { it < durationMs * 1_000L }
        .toList()

/** Chooses the eye direction whose positive perpendicular points toward mouth/nose. */
internal fun shouldKeepEyeOrder(
    firstX: Float,
    firstY: Float,
    secondX: Float,
    secondY: Float,
    downX: Float,
    downY: Float,
): Boolean {
    val eyeMidX = (firstX + secondX) * 0.5f
    val eyeMidY = (firstY + secondY) * 0.5f
    val cross = (secondX - firstX) * (downY - eyeMidY) -
        (secondY - firstY) * (downX - eyeMidX)
    return cross >= 0f
}

private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float =
    a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()

/** Suppresses tiny faces embedded in app chrome while retaining the dominant face(s). */
internal fun shouldKeepDetectedFace(faceWidth: Int, largestWidth: Int, frameWidth: Int): Boolean {
    if (faceWidth <= 0 || largestWidth <= 0 || frameWidth <= 0) return false
    val absoluteMinimum = frameWidth * 0.015f
    val relativeMinimum = largestWidth * 0.10f
    return faceWidth >= minOf(absoluteMinimum, relativeMinimum)
}

/** Eye landmark order can be reversed; choose the equivalent smallest roll correction. */
internal fun normalizedEyeRollDegrees(angle: Double): Double {
    var normalized = angle % 180.0
    if (normalized > 90.0) normalized -= 180.0
    if (normalized < -90.0) normalized += 180.0
    return normalized
}
