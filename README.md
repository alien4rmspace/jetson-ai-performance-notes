# Jetson AI performance notes

Measured optimization work from a camera, computer-vision, and local-LLM stack on an NVIDIA Jetson Orin Nano. This repository is a public portfolio record of the experiments, including changes that helped, tradeoffs, and failures that informed later designs.

The measurements are from a single device and workload. Stream FPS includes camera capture, inference, hand and pose processing, overlays, and encoding where those stages were enabled. TensorRT `trtexec` numbers measure an engine in isolation; they are not end-to-end stream FPS. Many experiments were short operational samples rather than repeated controlled trials, so the notes identify the test context instead of treating every difference as a general speedup.

## Results at a glance

| Experiment | Observation | What changed |
| --- | --- | --- |
| YOLOv8n stream inference | 8.5 FPS on CPU, 23.7 FPS on PyTorch CUDA, 30.0 FPS on TensorRT FP16 | Moved inference to the Orin GPU, then exported an FP16 engine. The 30 FPS camera ceiling limits the measured stream speedup. |
| Browser frame preparation | 3.10 ms to 1.61 ms for CPU frame mapping/color preparation | Negotiated BGR output with GPU `nvvideoconvert`, removing a CPU `cv2.cvtColor` step. This shifted work toward the GPU. |
| Local Qwen one-shot response | 14–18 s to 6.44 s for one conversational prompt; 14.3 s to 4.52 s for one command prompt | Built `llama-cpp-python` with CUDA and offloaded supported model layers. These are prompt samples, not a latency distribution. |
| TensorRT builder level, pose | 146.8 to 157.0 queries/s, level 3 to 5 | Same FP16 ONNX model, Orin, TensorRT 10.16.2, and 10-second `trtexec` benchmark. |
| TensorRT builder level, detector | 155.4 to 169.2 queries/s, level 3 to 5 | Same comparison protocol; level 5 also took longer to build. |

The detailed [case studies](notes/case-studies.md) explain the system-level work. The [TensorRT builder comparison](notes/tensorrt-builder-comparison.md) records the September 2026 experiment. The [sanitized field log](notes/field-log.md) preserves the May experiments, and the [later stream log](notes/recent-stream-log.md) covers May–August stability work. Older commands in these logs document experiments and may no longer match the current application.

## Test platform

- NVIDIA Jetson Orin Nano developer kit, 8 GB class, Ampere GPU (compute capability 8.7).
- CSI camera through Argus/GStreamer; later pipeline uses DeepStream for detection and pose, MediaPipe for hand gestures, and an HTTP MJPEG browser stream.
- YOLO models exported to ONNX and TensorRT FP16. Local Qwen inference uses `llama.cpp` through `llama-cpp-python`.
- Power mode and clocks affect results. Where recorded, the experiments used `MAXN_SUPER` and `jetson_clocks`; the isolated TensorRT comparison was run on the same device with the same benchmark flags for both builder levels. Actual instantaneous clocks were not logged for every comparison.

## Reading the measurements

FPS at a camera or browser cap is a throughput ceiling, not proof of spare capacity. CPU percentages in the field log are Linux process percentages, where 100% is roughly one full core. GPU utilization averages can hide short, latency-sensitive inference bursts. Build-time choices and inference-time performance are separate: the builder level 5 plans were faster in the recorded `trtexec` runs, but took more time to compile.

The implementation, model weights, TensorRT plans, recordings, and device-specific configuration are intentionally outside this notes repository. Addresses and local account paths in the historical log were replaced with documentation examples.
