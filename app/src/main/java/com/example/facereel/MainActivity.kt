package com.example.facereel

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.facereel.camera.CameraRecorder
import com.example.facereel.databinding.ActivityMainBinding
import com.example.facereel.processing.FacePipeline
import com.example.facereel.processing.TfliteEmbeddingEngine
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicLong

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var camera: CameraRecorder
    private lateinit var embeddings: TfliteEmbeddingEngine
    private lateinit var pipeline: FacePipeline

    private val handler = Handler(Looper.getMainLooper())
    private val recordingToken = AtomicLong(0L)
    private var recordingStartedAtNanos = 0L
    private var allowFinalizeProcessing = true
    private var state = State.PREVIEW

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) bindCamera() else setError("Camera permission is needed to record a reel.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        embeddings = TfliteEmbeddingEngine(this)
        pipeline = FacePipeline(this, embeddings)
        camera = CameraRecorder(this, this, binding.cameraPreview)
        binding.recordButton.setOnClickListener { onRecordButtonPressed() }
        val retainedClip = File(cacheDir, "last-debug-capture.mp4")
        if (BuildConfig.DEBUG && intent.getBooleanExtra("reprocess_debug", false) && retainedClip.exists()) {
            process(retainedClip)
        } else {
            ensureCameraPermission()
        }
    }

    override fun onStop() {
        super.onStop()
        if (state == State.RECORDING) {
            recordingToken.incrementAndGet()
            allowFinalizeProcessing = false
            handler.removeCallbacksAndMessages(null)
            camera.stop()
        }
    }

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        camera.release()
        embeddings.close()
        super.onDestroy()
    }

    private fun ensureCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            bindCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindCamera() {
        setStatus("Frame everyone in view, then record.")
        camera.bind { setError("Could not start camera: ${it.message}") }
    }

    private fun onRecordButtonPressed() {
        when (state) {
            State.RECORDING -> camera.stop()
            State.PROCESSING -> Unit
            State.PREVIEW, State.RESULT -> startRecording()
        }
    }

    private fun startRecording() {
        binding.collageImage.visibility = View.GONE
        binding.processingProgress.visibility = View.GONE
        allowFinalizeProcessing = true
        state = State.RECORDING
        binding.recordButton.isEnabled = false
        binding.recordButton.text = "STOP"
        binding.recordButton.contentDescription = "Stop recording"
        setStatus(
            if (embeddings.isReady) "Recording — up to 20 seconds."
            else "Recording — collage generation needs a reviewed model.",
        )
        camera.start(
            onStarted = {
                recordingStartedAtNanos = System.nanoTime()
                val token = recordingToken.incrementAndGet()
                binding.recordButton.isEnabled = true
                updateCountdown(token)
                handler.postDelayed({
                    if (recordingToken.get() == token) camera.stop()
                }, FacePipeline.MAX_DURATION_MS)
            },
            onFinalized = { result ->
                handler.removeCallbacksAndMessages(null)
                result.fold(
                    onSuccess = {
                        if (allowFinalizeProcessing) process(it.file) else {
                            it.file.delete()
                            setError("Recording stopped when the app left the foreground.")
                        }
                    },
                    onFailure = { setError(it.message ?: "Recording failed.") },
                )
            },
        )
    }

    private fun updateCountdown(token: Long) {
        if (state != State.RECORDING || recordingToken.get() != token) return
        val elapsedMs = (System.nanoTime() - recordingStartedAtNanos) / 1_000_000L
        val remaining = (FacePipeline.MAX_DURATION_MS - elapsedMs).coerceAtLeast(0L)
        binding.countdownText.text = "%02d:%02d".format(remaining / 60_000, (remaining / 1_000) % 60)
        handler.postDelayed({ updateCountdown(token) }, 100L)
    }

    private fun process(clip: File) {
        state = State.PROCESSING
        binding.recordButton.isEnabled = false
        binding.recordButton.text = "…"
        binding.processingProgress.visibility = View.VISIBLE
        binding.processingProgress.progress = 0
        setStatus("Finding the best face in each group…")
        lifecycleScope.launch {
            when (val outcome = pipeline.process(clip) { progress ->
                runOnUiThread { binding.processingProgress.progress = progress }
            }) {
                is FacePipeline.Outcome.Success -> showResult(
                    outcome.collage,
                    outcome.uniqueFaceCount,
                    outcome.candidateCount,
                )
                FacePipeline.Outcome.ModelUnavailable -> setError("Recognition model missing. No collage was made.")
                FacePipeline.Outcome.NoFacesFound -> setError("No usable faces were found in this reel.")
                is FacePipeline.Outcome.Failure -> setError("Processing failed: ${outcome.error.message}")
            }
            val retainedClip = File(cacheDir, "last-debug-capture.mp4")
            if (BuildConfig.DEBUG && clip.absolutePath != retainedClip.absolutePath) {
                clip.copyTo(retainedClip, overwrite = true)
            }
            if (clip.absolutePath != retainedClip.absolutePath) clip.delete()
        }
    }

    private fun showResult(collage: File, faceCount: Int, candidateCount: Int) {
        binding.collageImage.setImageBitmap(BitmapFactory.decodeFile(collage.absolutePath))
        binding.collageImage.visibility = View.VISIBLE
        state = State.RESULT
        binding.recordButton.isEnabled = true
        binding.recordButton.text = "NEW"
        binding.recordButton.contentDescription = "Record a new reel"
        binding.processingProgress.visibility = View.GONE
        binding.countdownText.text = "00:20"
        setStatus("$faceCount unique ${if (faceCount == 1) "face" else "faces"} from $candidateCount detections.")
    }

    private fun setError(message: String) {
        state = State.PREVIEW
        binding.recordButton.isEnabled = true
        binding.recordButton.text = "REC"
        binding.recordButton.contentDescription = "Start recording"
        binding.processingProgress.visibility = View.GONE
        binding.countdownText.text = "00:20"
        setStatus(message)
    }

    private fun setStatus(message: String) {
        binding.statusText.text = message
    }

    private enum class State { PREVIEW, RECORDING, PROCESSING, RESULT }
}
