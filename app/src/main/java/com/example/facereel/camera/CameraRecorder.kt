package com.example.facereel.camera

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/** Owns CameraX binding and one active recording. All callbacks arrive on the main executor. */
class CameraRecorder(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
) {
    data class CompletedClip(val file: File, val durationNanos: Long)

    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private val recordingStopRequested = AtomicBoolean(false)
    private var cameraProvider: ProcessCameraProvider? = null
    private var activeRecording: Recording? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var startedAtNanos = 0L

    fun bind(onError: (Throwable) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            runCatching {
                val provider = future.get()
                cameraProvider = provider
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                val qualitySelector = QualitySelector.from(
                    Quality.HD,
                    FallbackStrategy.lowerQualityOrHigherThan(Quality.SD),
                )
                val recorder = Recorder.Builder().setQualitySelector(qualitySelector).build()
                videoCapture = VideoCapture.withOutput(recorder)
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    videoCapture,
                )
            }.onFailure(onError)
        }, mainExecutor)
    }

    fun start(
        onStarted: () -> Unit,
        onFinalized: (Result<CompletedClip>) -> Unit,
    ) {
        val capture = videoCapture ?: return onFinalized(Result.failure(
            IllegalStateException("Camera is not ready."),
        ))
        if (activeRecording != null) return

        val outputDirectory = File(context.cacheDir, "clips").apply { mkdirs() }
        val clip = File(outputDirectory, "face-reel-${System.currentTimeMillis()}.mp4")
        recordingStopRequested.set(false)
        activeRecording = capture.output
            .prepareRecording(context, FileOutputOptions.Builder(clip).build())
            .start(mainExecutor) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        startedAtNanos = System.nanoTime()
                        onStarted()
                    }
                    is VideoRecordEvent.Finalize -> {
                        activeRecording = null
                        val duration = (System.nanoTime() - startedAtNanos).coerceAtLeast(0L)
                        if (event.hasError()) {
                            clip.delete()
                            onFinalized(Result.failure(
                                IllegalStateException("Video recording failed (${event.error}).", event.cause),
                            ))
                        } else if (!clip.exists() || clip.length() == 0L) {
                            onFinalized(Result.failure(IllegalStateException("No usable video was created.")))
                        } else {
                            onFinalized(Result.success(CompletedClip(clip, duration)))
                        }
                    }
                }
            }
    }

    /** Idempotent: called by the timer, cancel UI, and lifecycle stop. */
    fun stop() {
        if (recordingStopRequested.compareAndSet(false, true)) {
            activeRecording?.stop()
        }
    }

    fun isRecording(): Boolean = activeRecording != null

    fun release() {
        stop()
        cameraProvider?.unbindAll()
        cameraProvider = null
        videoCapture = null
    }

}
