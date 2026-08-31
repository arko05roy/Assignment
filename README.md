# Face Reel

Face Reel records a maximum 20-second rear-camera clip, samples it at two frames per second, detects faces locally, ranks crop quality, creates embeddings, clusters recurring faces, and renders one best crop per cluster into a collage. The completed collage receives an app-scoped `FileProvider` URI; no broad media permission is needed.

## Project status

The app foundation and offline pipeline are implemented. A camera preview is bound to the activity, recordings are written only to `cacheDir/clips`, and the timer starts from CameraX's `Start` event. Recording stops exactly once at 20 seconds, or earlier if the user stops it. Processing begins only after CameraX finalizes a usable file.

Face detection and unique-face clustering both run locally. ML Kit provides detection boxes; MobileFaceNet embeddings, not ML Kit tracking IDs, determine whether two crops belong to one person.

## Recognition model

The app bundles `mobile_face_net.tflite`, a ~5 MB MobileFaceNet embedding model from the Apache-2.0 licensed [`hugocornellier/face_detection_tflite`](https://github.com/hugocornellier/face_detection_tflite) repository at commit `50c784adaa9f40c722affb1d4412674f25e1fe0c`. Its SHA-256 checksum and notice are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

At startup, the app validates a Float32 `[1, height, width, 3]` RGB input and a single Float32 embedding output. It applies `[-1, 1]` RGB normalization, serializes inference to one interpreter, L2-normalizes the embeddings, and geometrically aligns the eye centres to a fixed 112×112 MobileFaceNet input. It groups strong pairwise matches into connected components at a cosine-similarity threshold of `0.75`, calibrated against the included demo capture's measured same-identity and cross-identity distributions.

## Build and run

Use Android Studio with JDK 17 and Android SDK Platform 36, or run:

```sh
./gradlew assembleDebug
```

The project is configured with Android Gradle Plugin 9.2.0, Gradle 9.4.1, CameraX 1.6.1, and ML Kit face detection 16.1.7. No broad media-read or audio permission is requested.

## Mac demo

An Apple Silicon Pixel 7 emulator named `FaceReel_Demo` is configured on this Mac. It maps the emulator's rear camera to `webcam1`, the detected **MacBook Air Camera**. Run the demo from a normal Terminal window:

```sh
cd /Users/arkoroy/Desktop/mo
./scripts/run-mac-demo.sh
```

Keep that terminal open while presenting. If the preview is black, allow the emulator's camera access in **System Settings → Privacy & Security → Camera**, then close and rerun the emulator. Record the presentation itself with macOS's screen recorder (`⌘⇧5`), selecting the emulator window and microphone as needed.

## Pipeline

```text
temporary MP4 → 40 scaled frames → ML Kit bounds → padded/clamped crops
→ quality score → TFLite embeddings → centroid clustering → JPEG collage
```

The crop rules reject anything under 100×100 pixels and use 20% clamped padding. Quality combines crop area (45%), Laplacian-variance sharpness (35%), and head-pose frontalness (20%).
