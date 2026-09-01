package com.example.facereel.processing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * MobileFaceNet-compatible interpreter.
 *
 * This implementation intentionally only enables when a reviewed model named
 * `mobile_face_net.tflite` is supplied in assets. The app must not silently substitute
 * tracking IDs or detector output for identity embeddings.
 */
interface EmbeddingEngine : AutoCloseable {
    val isReady: Boolean
    suspend fun embed(face: Bitmap): FloatArray
}

class TfliteEmbeddingEngine(context: Context) : EmbeddingEngine {
    private val interpreter: Interpreter?
    private val interpreterLock = Any()
    private val inputWidth: Int
    private val inputHeight: Int
    private val outputSize: Int

    override val isReady: Boolean get() = interpreter != null

    init {
        val loaded = runCatching {
            val model = context.assets.open("facenet.tflite").use { input ->
                input.readBytes().let { ByteBuffer.allocateDirect(it.size).order(ByteOrder.nativeOrder()).put(it).apply { rewind() } }
            }
            Interpreter(model, Interpreter.Options().setNumThreads(1))
        }.onFailure { Log.e(TAG, "Could not load the bundled face embedding model.", it) }.getOrNull()
        interpreter = loaded
        if (loaded == null) {
            inputWidth = 0
            inputHeight = 0
            outputSize = 0
        } else {
            val inputShape = loaded.getInputTensor(0).shape() // expected [1, height, width, 3]
            val outputShape = loaded.getOutputTensor(0).shape()
            require(loaded.getInputTensor(0).dataType() == DataType.FLOAT32) {
                "The supplied model must accept Float32 input."
            }
            require(loaded.getOutputTensor(0).dataType() == DataType.FLOAT32) {
                "The supplied model must return Float32 embeddings."
            }
            require(inputShape.size == 4 && inputShape[0] == 1 && inputShape[3] == 3) {
                "The supplied model must accept a [1, height, width, 3] RGB tensor."
            }
            require(outputShape.size == 2 && outputShape[0] == 1) {
                "The supplied model must return one embedding vector."
            }
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
            outputSize = outputShape[1]
            Log.i(TAG, "Loaded MobileFaceNet: ${inputWidth}x$inputHeight RGB → $outputSize dimensions")
        }
    }

    override suspend fun embed(face: Bitmap): FloatArray = synchronized(interpreterLock) {
        val runner = checkNotNull(interpreter) { "Face recognition model is unavailable." }
        val resized = Bitmap.createScaledBitmap(face, inputWidth, inputHeight, true)
        val input = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until inputHeight) for (x in 0 until inputWidth) {
            val pixel = resized.getPixel(x, y)
            // RGB values normalized to [-1, 1]. Confirm this against the supplied model card.
            input.putFloat((((pixel shr 16) and 0xff) - 127.5f) / 127.5f)
            input.putFloat((((pixel shr 8) and 0xff) - 127.5f) / 127.5f)
            input.putFloat(((pixel and 0xff) - 127.5f) / 127.5f)
        }
        input.rewind()
        if (resized !== face) resized.recycle()
        val output = Array(1) { FloatArray(outputSize) }
        runner.run(input, output)
        return output[0].l2Normalized()
    }

    override fun close() = interpreter?.close() ?: Unit

    private companion object {
        const val TAG = "FaceEmbedding"
    }
}

private fun FloatArray.l2Normalized(): FloatArray {
    val magnitude = sqrt(sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-12f)
    return FloatArray(size) { this[it] / magnitude }
}
