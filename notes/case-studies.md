# Optimization case studies

## 1. Move detection from CPU to TensorRT

The initial YOLOv8n camera stream measured 8.5 FPS on CPU. Running PyTorch on the Orin GPU measured 23.7 FPS. An FP16 TensorRT engine measured 30.0 FPS in the same application path, reaching the camera's nominal 30 FPS rate. That is a 3.5× observed stream gain over the CPU configuration, with the important limit that the camera ceiling hides any additional engine headroom. The historical model used a 320-pixel inference size. The [field log](field-log.md) has the original configuration and process samples.

Later changes added direct TensorRT detector execution and capped browser JPEG encoding to its own publish rate so inference frames did not all pay the browser cost. Pose and hand recognition were scheduled separately. This kept the stream near 30 FPS in the recorded samples while adding gesture capability, but hand recognition still consumed significant CPU when active.

## 2. Keep the camera pipeline in DeepStream

The later path used Argus capture, NVMM buffers, DeepStream object detection, and a secondary pose inference engine on person boxes. Pose keypoints came from DeepStream tensor metadata instead of a separate CPU-frame pose path. A DeepStream object-only HTTP sample held 30 FPS at about 26.6% process CPU and 364 MB RSS. Adding hands, pose, and command handling increased CPU and memory; those samples should not be compared as if the workloads were identical.

Moving hand landmarks to a TensorRT path removed the separate MediaPipe worker in one test, but the CPU MediaPipe worker remained preferable for closed-fist recognition quality in later operation. That choice illustrates the practical tradeoff: lower resource use was not enough if command recognition became less reliable.

## 3. Remove avoidable browser-frame work

The HTTP path initially encoded a JPEG, decoded it back to a CPU BGR frame for hand processing, and encoded it again for the browser. Replacing that round trip with a raw BGRx appsink removed an avoidable JPEG decode. Capping browser publication at 20 FPS while keeping camera/inference at roughly 30 FPS lowered a sampled main-process CPU reading from roughly 88–98% to about 80%; the hottest image-processing thread fell from about 55.5% to 38.1% CPU in the recorded samples.

For a 4K camera source with a 1080p CPU hand path, direct BGR through default VIC conversion failed format negotiation. Selecting GPU `nvvideoconvert` made BGR work and eliminated a CPU BGRx-to-BGR conversion. The measured CPU frame map/color step changed from about 3.10 ms to 1.61 ms. The short live sample returned to about 30 FPS, while power and GPU load increased. The notes flag stability concerns seen with GPU copy paths on an older JetPack/DeepStream stack; this was a workload-specific tradeoff, not a universal default.

## 4. Isolate and offload the local LLM

Building `llama-cpp-python` with CUDA allowed supported Qwen layers to run on the GPU. In two one-shot prompt samples, response time changed from about 14–18 s CPU-only to 6.44 s with offload for a conversational prompt, and from 14.3 s to 4.52 s for a command prompt. Peak resident memory also fell in those samples from about 2.0 GB to 1.30 GB. Prompt and process startup effects are included in these one-shot measurements.

Loading Qwen in the DeepStream process destabilized the CUDA/video pipeline during an early test. A separate warm local LLM service kept the model resident without sharing the DeepStream process. One later HTTP request through the warm service measured 1,653 ms. The warm service added about 607 MB idle RSS in that recorded state, trading memory for lower response latency and process isolation.

## 5. Diagnose a GPU-copy crash before tuning engines

On JetPack 6.2 with DeepStream 7.1, recurring stream exits showed `nvbufsurftransform_copy.cpp` copy failures, `nvgpu` MMU faults, and CUDA error 700. The detector and pose plans also had a device/profile mismatch warning, so they were rebuilt locally. A rebuilt plan still hit the same copy failure. Changing the affected `nvvideoconvert` copy path to VIC (`copy-hw=2`) addressed the observed crash signature on that stack; the stream then held about 30 FPS through the recorded validation window without a new fault. The engine warning was a separate deployment risk, not the root cause of that crash.

That workaround is tied to the older software stack and should be retested on other JetPack/DeepStream versions. The diagnostic lesson was to correlate application errors with kernel GPU faults and buffer-transform logs before attributing a failure to Python, model performance, memory pressure, or temperature.

## 6. Compare TensorRT builder levels with measured plans

The September 2026 experiment built separate FP16 detector and pose plans at builder levels 3 and 5, then loaded and benchmarked each plan for 10 seconds after a one-second warm-up. Level 5 improved isolated throughput by 6.9% for pose and 8.9% for detection in those runs. The [full comparison](tensorrt-builder-comparison.md) records GPU compute time and one-time build cost. Both level 5 plans were selected for the local deployment; a full stream test remains a separate validation step.

## 7. Standardize RGB and remove compressed hand-frame transfer

In the September 28 pipeline, DeepStream supplied RGB frames that the application converted to BGR. Eligible hand crops were JPEG-encoded in the camera callback, sent to the MediaPipe subprocess, decoded, and converted back to RGB. Pending encoded crops could be replaced before inference consumed them.

The replacement keeps live frames in RGB, retains the latest owned crop, and prepares raw shared memory when the hand worker is ready. Browser encoding uses an RGB-compatible JPEG interface; recording accepts RGB through GStreamer. Two approximately 30-second Nsight captures showed frame-callback p95 falling from 16.66 to 13.25 ms and main-process CPU falling from 21.77% to 19.68% of the six-core CPU. The hand-submission marker's median fell from 2.77 to 0.38 ms, although the proportion of calls eligible to prepare a frame differed.

MediaPipe recognition showed no consistent latency improvement. Its median rose from 77.28 to 83.03 ms, while p95 fell from 122.33 to 119.92 ms. The captures did not contain scheduling data for the separate workers, and the recording encoder changed as part of the migration. The supported conclusion is reduced camera-callback cost and observed main-process CPU usage; total-system savings and causal attribution require a controlled comparison. The [full note and derived data](rgb-shared-memory-comparison.md) retain the measurement scopes and sample counts.

## 8. Remove the second CPU copy when receiving GPU frames

The RGB pipeline still copied each GPU frame into a temporary host buffer and then copied that buffer into an owned NumPy array. I allocated the final array first, copied contiguous tensors directly into it, and used `cudaMemcpy2D` for RGB surfaces with padded rows. The result still owns its memory after the source buffer is released. Unusual tensor layouts retain a fallback that uses a temporary buffer.

In a later pair of 30-second captures, mean RGB frame-copy time fell from 6.17 to 3.16 ms. Frame-callback p50/p95/p99 changed from 9.63/19.09/29.67 ms to 7.22/13.37/17.13 ms, and main-process CPU across six cores fell from 28.06% to 23.78%. The new report recorded 900 2D CUDA copies. Tests checked padded and contiguous layouts, byte offsets, ownership, and errors, including real CUDA transfers.

These observations cover the main process and synchronous callback. Separate worker CPU and camera-to-browser latency were not measured, and scene/PTZ conditions were not controlled. The [full comparison and derived data](direct-numpy-copy-comparison.md) preserve the evidence and calculation details.
