# Direct-to-NumPy frame-copy comparison

Date: 2026-09-28 (PDT)

## Change and scope

The frame-copy helper previously allocated a temporary CPU buffer, copied the
GPU tensor into it, and then called NumPy `.copy()` to produce the owned output.
The updated helper allocates the final NumPy array first:

```text
Before: GPU frame -> temporary CPU buffer -> owned NumPy array
After:  GPU frame -> owned NumPy array
```

Contiguous CUDA tensors use `cudaMemcpy` directly into the array. RGB images
with padded rows use `cudaMemcpy2D`, passing the source row pitch and the compact
NumPy row pitch. CPU tensors use a source view followed by one owned copy.
Arbitrary CUDA layouts, such as transposed or column-strided tensors, retain a
temporary-buffer fallback. The source DLPack capsule remains alive until the
synchronous copy finishes.

This removes an intermediate allocation and CPU-to-CPU copy on the image path.
A device-to-host copy remains. The helper also reads pose output tensors, so its
changes are not limited to full-frame images. The implementation passed 27
relevant tests, including real CUDA transfers for contiguous and padded inputs.

## Captures

Both runs already used RGB frames and shared-memory hand transfer, with
`--cpu-frame-processing` enabled. GPU metric sampling was 1,000 Hz and Python
sampling was 100 Hz in both reports. This pair has its own baseline and should
not be combined with the earlier BGR/JPEG comparison to calculate a cumulative
percentage improvement.

| Run | Report | Duration | Main stream PID |
| --- | --- | ---: | ---: |
| Before: temporary buffer + NumPy copy | `vision-20260928-182731-cpu-frames.nsys-rep` | 30.000 s | 23279 |
| After: direct NumPy copy | `vision-20260928-185712-cpu-frames.nsys-rep` | 30.000 s | 30047 |

## Latency

N is the number of completed NVTX ranges wholly inside the capture window.
All durations below are milliseconds. p50 is the median; p95/p99 are the
95th/99th percentiles of those durations.

| Marker | Run | N | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Frame Processing` | Before | 899 | 10.82 | 9.63 | 19.09 | 29.67 |
| `Frame Processing` | After | 900 | 7.69 | 7.22 | 13.37 | 17.13 |
| `RGB Frame Copy` | Before | 899 | 6.17 | 5.25 | 11.79 | 16.30 |
| `RGB Frame Copy` | After | 900 | 3.16 | 2.74 | 5.91 | 9.01 |
| `Process CPU Frame` | Before | 899 | 2.47 | 2.10 | 7.08 | 9.94 |
| `Process CPU Frame` | After | 900 | 2.25 | 2.14 | 5.04 | 8.39 |
| `Process Hand Frame` | Before | 449 | 0.50 | 0.40 | 0.89 | 2.49 |
| `Process Hand Frame` | After | 447 | 0.55 | 0.50 | 0.76 | 1.59 |
| `MediaPipe Hand Inference` | Before | 311 | 94.03 | 90.06 | 122.37 | 144.68 |
| `MediaPipe Hand Inference` | After | 350 | 82.48 | 86.29 | 98.34 | 106.41 |

`Frame Processing` measures the synchronous camera callback, excluding upstream
YOLO work and completion of asynchronous MediaPipe, browser, and recording work.
It is not camera-to-browser latency. `RGB Frame Copy` includes buffer extraction,
allocation, and the tensor-copy helper; it is not just CUDA transfer duration.
`Process CPU Frame` covers the command/overlay call within the callback.
`Process Hand Frame` covers hand submission, including calls that skip input
preparation. `MediaPipe Hand Inference` covers the synchronous recognizer call
and its internal waits in a separate process. Nested ranges overlap and must
not be summed to estimate total latency.

Observed frame-callback p50/p95/p99 decreased by **25.0% / 30.0% / 42.3%**.
Mean frame-copy time decreased by **48.8%**, from **6.17 to 3.16 ms**.
MediaPipe timings also changed, but the input and scheduling conditions were
not controlled; this does not isolate an improvement in the model itself.

## Average CPU utilization

| Run | Scheduled CPU time | Average busy cores | CPU utilization across six cores |
| --- | ---: | ---: | ---: |
| Before | 50.502 s | 1.683 | 28.06% |
| After | 42.803 s | 1.427 | 23.78% |

Main-process CPU fell by **4.28 percentage points**, or **15.2% relative**.
Scheduling coverage contains only the main stream process in each report. The
separate MediaPipe, browser, recording, and profiler processes are excluded, so
this does not establish total application or whole-system CPU savings.

## Workload and copy-path evidence

| Recorded activity | Before | After |
| --- | ---: | ---: |
| Completed detector `enqueueV3` calls | 898 | 899 |
| Completed pose `enqueueV3` calls | 898 | 900 |
| Completed MediaPipe recognition calls | 311 | 350 |
| `cudaMemcpy2D` API calls | 0 | 900 |

Pose threads were identified from keypoint-layer NVTX labels. The remaining
TensorRT enqueue thread serves the detector. Enqueue counts describe CPU
submission calls, not measured GPU inference durations.

The 900 recorded 2D copies confirm the new frame-copy path ran. Similar detector
and pose counts help describe the workload, but do not establish identical
images, ROIs, or scheduling. MediaPipe processed more frames in the later run.

## Evidence and calculation

The original Nsight reports and SQLite exports remain local. The published
[derived comparison data](../data/2026-09-28-direct-numpy/comparison.json) include
unrounded latencies, sample counts, CPU totals, boundary handling, inference
counts, and copy API counts.

Latency is `(end - start) / 1e6`, using named ranges in `NVTX_EVENTS` and
`StringIds`. Only completed ranges with `0 <= start < end <= 30 seconds` are
included. Percentiles use NumPy's linear interpolation.

CPU time comes from matching `SCHED_EVENTS` schedule-in/out events per thread
and CPU. Intervals are clipped to the capture window from `RUN_DURATION_MS`.
Four intervals still open at the end of the before capture are counted through
30 seconds. One initial schedule-out in the after capture is counted from time
zero. No other pairing anomalies were found. `CpuCores` is six in both exports.

```text
average busy cores = total scheduled CPU seconds / 30
six-core CPU utilization (%) = average busy cores / 6 * 100
relative decrease (%) = (before - after) / before * 100
```

## Limits

- There is one capture per version; scene and PTZ conditions were not controlled.
  The local PTZ configuration was also edited during this work, and exact
  per-capture configuration snapshots were not preserved.
- These captures measure the application under profiling. They do not establish
  uninstrumented performance or guarantee the same improvement on another scene.
- Both reports warn that some CUDA and NVTX events may be missing. Counts and
  percentiles describe the recorded, valid ranges.
- The callback measurement excludes background completion, and the CPU
  measurement excludes separate workers. A controlled replay with broader
  scheduling coverage is needed to attribute total application savings.
