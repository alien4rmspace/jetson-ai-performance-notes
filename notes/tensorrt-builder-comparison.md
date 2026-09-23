# TensorRT builder optimization level 3 versus 5

Measured on 2026-09-22 with an NVIDIA Jetson Orin Nano and TensorRT 10.16.2. Both models were static batch-one YOLO26s ONNX exports built as FP16-capable TensorRT plans with a 3,071 MiB workspace setting. Each builder level used its own timing cache. `trtexec` loaded each plan and measured inference for at least 10 seconds after a 1,000 ms warm-up, with data transfers enabled. The same GPU and benchmark options were used for all four plans.

| Model | Builder level | Build time | Throughput | Mean latency | Mean GPU compute | P99 GPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pose | 3 | 440.30 s | 146.832 qps | 7.198 ms | 6.804 ms | 7.085 ms |
| Pose | 5 | 1,101.53 s | 156.991 qps | 6.791 ms | 6.364 ms | 6.543 ms |
| Detector | 3 | 38.34 s | 155.401 qps | 6.824 ms | 6.429 ms | 6.714 ms |
| Detector | 5 | 175.52 s | 169.218 qps | 6.312 ms | 5.904 ms | 6.184 ms |

Level 5 improved throughput by **6.9% for pose** and **8.9% for detection** relative to level 3. Mean GPU compute time fell by about 6.5% and 8.2%, respectively. The one-time build cost increased about 2.5× for pose and 4.6× for detection. All four builds and benchmark runs completed successfully. The build logs contained a weakly typed network deprecation warning and an expected first-use timing-cache miss; no benchmark error was recorded.

The benchmark ran each engine alone. It does not include camera capture, DeepStream scheduling, MediaPipe, overlay work, browser encoding, or other concurrent GPU work. Results are a single run per plan, and actual instantaneous GPU clocks were not captured, so a repeated, alternating-order benchmark would improve confidence if a small difference mattered. The local deployment selected level 5 for both plans, with end-to-end stream validation treated separately.

Representative commands used the following form:

```bash
trtexec --onnx=model.onnx --saveEngine=model_opt5.engine \
  --fp16 --builderOptimizationLevel=5 --memPoolSize=workspace:3071 \
  --timingCacheFile=timing-opt5.cache --skipInference

trtexec --loadEngine=model_opt5.engine --warmUp=1000 --duration=10
```

The original generated engines and logs remain local to the Jetson; they are device-specific binary artifacts and are not distributed here.
