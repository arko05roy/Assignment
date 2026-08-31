
This separation makes the recognition pipeline unit-testable without a camera and makes CameraX lifecycle work isolated from ML work.

## 6. Camera recording implementation

### Permissions

```xml
<uses-permission android:name="android.permission.CAMERA" />
<!-- Include only if sound is deliberately recorded. -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Do not ask for broad media-read permissions: the app records its own video and creates its own collage.

### Bind the use cases

1. Obtain a `ProcessCameraProvider`.
2. Create `Preview`, supplying its surface to `PreviewView`.
3. Create `Recorder` with a conservative `QualitySelector` such as HD/FHD with fallback.
4. Create `VideoCapture.withOutput(recorder)`.
5. Bind preview and capture to the activity lifecycle with the rear lens selector.
6. Rebind when changing cameras, only while not recording.

Use a temporary file in `context.cacheDir` or app-specific external files directory. A local temporary target makes cancel/cleanup straightforward and avoids exposing incomplete footage to Gallery.

### Exact 20-second stop policy

- Define `MAX_DURATION_MS = 20_000L` in one location.
- Record the monotonic start timestamp when CameraX emits its start event, not merely when the user taps the button.
- Update the countdown from elapsed monotonic time.
- Schedule `recording.stop()` for the deadline; use an `AtomicBoolean`/mutex to ensure it happens once.
- Treat CameraX `Finalize` as the point at which the video URI/path becomes processable.
- If early finalization occurs, process the completed duration if valid, but indicate failure when no usable video exists.

### Camera lifecycle rules

- On `onStop`, stop an active recording and invalidate the recording token; do not continue camera capture in the background for this assignment.
- On rotation, preserve the ViewModel state and rebind preview. If recording does not safely survive a configuration change on the chosen CameraX version, lock orientation while recording; document the decision.
- Never rebind use cases mid-recording unless the chosen CameraX behavior has been specifically verified.
- Make cancel and finalization idempotent.

## 7. Video-to-face processing pipeline

### Pipeline overview

```text
temporary MP4
  → read duration / rotation metadata
  → sample 40 scaled frames
  → ML Kit bounding boxes
  → crop + pad + clamp
  → reject bad crops and score quality
  → normalize / embed with TFLite
  → cluster by cosine similarity
  → choose highest-quality representative per cluster
  → render collage bitmap
  → save output URI
```

### 7.1 Frame sampling

Sampling 2 frames/second creates 40 timestamps over a 20-second clip. This is adequate for an assignment demo, gives each moving person multiple opportunities to be seen, and avoids decoding every video frame.

```kotlin
val sampleIntervalUs = 500_000L
val durationUs = min(retrievedDurationMs, 20_000L) * 1_000L
val timestamps = generateSequence(0L) { it + sampleIntervalUs }
    .takeWhile { it < durationUs }
    .toList()
```

For each timestamp:

1. call `getScaledFrameAtTime()` close to that time;
2. apply video rotation metadata if needed before detection/cropping;
3. recycle or release frame/crop bitmaps as soon as downstream work is complete;
4. emit progress as `frameIndex / totalFrames`.

Use `OPTION_CLOSEST` for coverage if performance permits. It provides a timestamp-close frame but has more overhead than sync-only extraction. Fall back to a sync-frame option if a device proves too slow.

### 7.2 ML Kit detector configuration

Initial configuration:

```kotlin
FaceDetectorOptions.Builder()
    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
    .setMinFaceSize(0.10f)
    .enableTracking()
    .build()
```

Tracking IDs are useful as a weak, intra-video hint but **must not be the identity solution**. They can change after occlusion or disappear across frames. Embeddings remain the source of truth.

### 7.3 Face crop rules

For every ML Kit `Face`:

1. read its bounding rectangle;
2. add approximately 15–25% symmetric padding so the recognition crop includes the full face boundary;
3. clamp the rectangle to the bitmap bounds;
4. reject crops smaller than 100×100 pixels after clamping;
5. retain frame timestamp, detection tracking ID, pose angles, crop dimensions, and quality measurements.

Avoid using an out-of-bounds rectangle directly: a partial face close to the frame edge is common and must not crash the pipeline.

### 7.4 Candidate and quality data

```kotlin
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
    val qualityScore: Double
)
```

Start with a deterministic quality score:

```text
quality = 0.45 × normalizedFaceArea
        + 0.35 × normalizedSharpness
        + 0.20 × frontalPoseScore
```

- `normalizedFaceArea`: crop area relative to sampled-frame area, capped at 1.
- `normalizedSharpness`: normalized variance of a Laplacian applied to a small grayscale image; use it as a ranking signal, not as an absolute truth.
- `frontalPoseScore`: penalize absolute yaw and roll. For example, a nearly frontal crop scores higher than one rotated 30°.

The best-quality candidate, not the latest candidate, is used as the final cluster representative.

## 8. Face embedding and identity clustering

### 8.1 Recognition model selection gate

The project cannot honestly meet its uniqueness requirement without a known embedding model. Before coding the recognizer, confirm:

- Input resolution (commonly 112×112 or 160×160).
- RGB channel order.
- Value range and normalization formula (for example `[0, 255] → [-1, 1]`).
- Need for face alignment and supported landmark inputs.
- Output vector dimension (commonly 128 or 512).
- Whether the output requires L2 normalization.
- Runtime device performance.
- Model weights’ license, attribution, and permitted use.

Use MobileFaceNet or a similarly small ArcFace/FaceNet-compatible embedding model only after verifying these properties from its original source. The README must identify the model precisely rather than simply saying “MobileFaceNet.”

### 8.2 Preprocessing

For each accepted crop:

1. Convert to the model’s expected aspect ratio.
2. Align the crop based on eye landmarks if the model requires aligned faces. If no landmark/alignment implementation is available in the first milestone, use padded center-crop resize and document the limitation.
3. Resize using high-quality bilinear sampling to the model input resolution.
4. Convert to the expected RGB tensor format and normalization range.
5. Invoke exactly one TFLite interpreter at a time unless the model implementation is proven thread-safe.
6. L2-normalize the returned vector:

```text
e_normalized = e / sqrt(sum(e_i²) + 1e-12)
```

### 8.3 Cosine similarity

For normalized vectors, cosine similarity is their dot product:

```text
similarity(a, b) = Σ(a_i × b_i)
```

It returns values near `1.0` for strongly similar embeddings and lower values for different faces. The model and preprocessing determine meaningful values, so the exact threshold is empirical.

### 8.4 Incremental clustering algorithm

Start with a simple, explainable centroid clustering approach:

```text
for each candidate, ordered by descending quality:
    calculate cosine similarity against each cluster centroid
    if best similarity >= threshold: