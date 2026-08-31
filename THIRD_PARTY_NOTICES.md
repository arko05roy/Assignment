# Third-party notices

## MobileFaceNet embedding model

- File: `app/src/main/assets/mobile_face_net.tflite`
- Source: `hugocornellier/face_detection_tflite`, commit `50c784adaa9f40c722affb1d4412674f25e1fe0c`
- Upstream path: `assets/models/mobilefacenet.tflite`
- Model family: MobileFaceNets, based on the 2018 MobileFaceNets paper
- License: Apache License 2.0 (upstream repository license)
- SHA-256: `be4bc7cfc53f7bc336d0f28b1ab92535f618c913a422b683210750f6b5354854`

The app validates the input and output tensor shape/type at runtime and runs this model entirely on
device. It never sends frames, crops, or embeddings over the network.
