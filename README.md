# Optimizing real-time AI on Jetson Orin Nano

I profiled and optimized a live camera, computer-vision, and local-LLM stack on an NVIDIA Jetson Orin Nano. Moving YOLOv8n inference from CPU to TensorRT FP16 raised the measured camera stream from **8.5 to 30.0 FPS** (the camera's 30 FPS limit). Later work moved detection and pose into DeepStream, reduced browser-frame CPU work, and isolated local Qwen inference from the video process.

This repository is my performance engineering portfolio: the results and decisions are here in the README; the linked notes preserve the measurements and failure investigations behind them.

## Measured results

| Problem | Change I made | Observed result |
| --- | --- | --- |
| CPU inference limited the YOLOv8n stream | Moved inference to PyTorch CUDA, then exported a TensorRT FP16 engine | **8.5 → 23.7 → 30.0 FPS** in the application stream; 30 FPS was the camera ceiling. |
| CPU color conversion added cost to the browser/hand path | Negotiated BGR output through GPU `nvvideoconvert` and removed `cv2.cvtColor` | **3.10 → 1.61 ms** for CPU frame mapping/color preparation in short live samples. |
| TensorRT builder settings had an unknown payoff | Built separate YOLO26s detector and pose plans at optimization levels 3 and 5, then benchmarked each | Level 5 gave **+8.9% detector** and **+6.9% pose** throughput in isolated 10-second `trtexec` runs. |
| CPU-only local Qwen responses were slow | Built `llama-cpp-python` with CUDA and offloaded supported layers | One command prompt fell from **14.3 → 4.52 s**; a conversational sample fell from **14–18 → 6.44 s**. |

## TensorRT optimization level 3 vs 5

I created a benchmarking script that automatically builds and compares FP16-capable YOLO26s pose and detector TensorRT plans across builder optimization levels. On the same Orin Nano with TensorRT 10.16.2, the script built level 3 and level 5 plans, ran each through trtexec for 10 seconds after a 1-second warm-up, collected latency and throughput metrics, and selected the faster plans locally. Build time is a one-time cost; the other columns measure isolated engine inference.

| Model | Level | Throughput (qps) | Mean latency (ms) | GPU mean / p99 (ms) | Build time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pose | 3 | 146.832 | 7.198 | 6.804 / 7.085 | 440.30 |
| Pose | **5** | **156.991** | **6.791** | **6.364 / 6.543** | 1,101.53 |
| Detector | 3 | 155.401 | 6.824 | 6.429 / 6.714 | 38.34 |
| Detector | **5** | **169.218** | **6.312** | **5.904 / 6.184** | 175.52 |

Level 5 increased throughput by 6.9% for pose (146.8 → 157.0 qps) while reducing mean latency by 0.407 ms (5.7%). For detection, throughput increased by 8.9% (155.4 → 169.2 qps) while mean latency decreased by 0.512 ms (7.5%). GPU mean inference time improved by 0.440 ms (6.5%) for pose and 0.525 ms (8.2%) for detection. These gains came at approximately 2.5× and 4.6× longer one-time build times, respectively. Both level 5 plans were selected by the comparison script. The [full benchmark note](notes/tensorrt-builder-comparison.md) records the configuration and limits.

## System and engineering decisions

The production path grew from a Python YOLO camera stream into **CSI camera → Argus/GStreamer → DeepStream YOLO26s detection and pose → hand-gesture worker → browser MJPEG**. I decoupled browser publication from camera/inference cadence, removed a JPEG encode/decode round trip before CPU hand processing, and used a secondary DeepStream pose engine on person regions. I kept MediaPipe for hands when it recognized closed-fist gestures more reliably than a lower-CPU TensorRT hand path.

An intermittent video crash was a separate performance and reliability problem. I correlated DeepStream buffer-copy failures with kernel GPU MMU faults and CUDA error 700. Rebuilding the TensorRT engines fixed a device/profile warning but did **not** stop the crash. On the JetPack 6.2 / DeepStream 7.1 stack, switching the affected `nvvideoconvert` copies to VIC (`copy-hw=2`) addressed the observed failure; the stream held about 30 FPS through the recorded validation window. That workaround is specific to the older stack, not a blanket recommendation for every Jetson setup.

For voice, I moved Qwen into a separate warm service after loading it inside the DeepStream process destabilized the pipeline. That traded roughly 607 MB of idle resident memory in one sample for process isolation and faster repeated responses.

## Evidence and limits

- **Hardware:** Jetson Orin Nano developer kit, 8 GB class, Ampere GPU (compute capability 8.7); CSI camera; YOLO, TensorRT, DeepStream, MediaPipe, and local Qwen/`llama.cpp`.
- **Measurement context:** Stream FPS includes the enabled camera, inference, hand/pose, overlay, and encoding stages. The 30 FPS camera cap limits what a stream FPS result can show. The builder-level comparison measures each TensorRT engine alone, not the full DeepStream pipeline.
- **Confidence:** These are measurements from one device. Many are short operational samples rather than repeated controlled trials. Power mode and clocks can affect comparisons; instantaneous GPU clocks were not recorded for every run.

Read the [case studies](notes/case-studies.md) for the changes and tradeoffs, the [TensorRT level 3 versus 5 comparison](notes/tensorrt-builder-comparison.md) for full benchmark numbers, or the sanitized [May field log](notes/field-log.md) and [later stream log](notes/recent-stream-log.md) for the chronological record. Commands in the logs document the configuration at the time and may no longer match the current application.

The production application, model weights, TensorRT plans, recordings, and device-specific configuration are outside this notes repository. Local account paths and LAN addresses in the historical logs were replaced with documentation examples.
