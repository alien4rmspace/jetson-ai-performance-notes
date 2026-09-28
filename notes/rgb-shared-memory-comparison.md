# RGB and shared-memory stream comparison

Date: 2026-09-28 (PDT)

Two Nsight Systems captures on the six-core Jetson Orin Nano compare the
previous BGR/JPEG hand-transfer path with the RGB/shared-memory implementation.
These are observed application runs, not repeated trials using identical frames.

## Captures and changes

| Label | Report | Capture duration | Stream PID | MediaPipe PID |
| --- | --- | ---: | ---: | ---: |
| Before | `vision-20260928-123919.nsys-rep` | 29.999 s | 19813 | 19846 |
| After | `vision-20260928-130901.nsys-rep` | 30.000 s | 22406 | 22472 |

The updated application keeps CPU frames in RGB, replaces JPEG transfer to
MediaPipe with raw RGB shared memory, and resizes the selected pending crop when
the worker is ready. Browser JPEG encoding uses an RGB-compatible Pillow
interface. Recording accepts RGB through GStreamer and writes software H.264
MP4, with an MJPEG/AVI fallback. Color-dependent tracking, focus, exposure, and
overlay operations were updated with the same convention.

Multiple components changed together, including the recording encoder. The
results cannot isolate the effect of a single change or establish total CPU
savings for the entire application.

## Latency of comparable NVTX ranges

All times are milliseconds. p50 is the median; p95/p99 are the duration thresholds
below which 95%/99% of measured calls fall. N counts completed ranges.

| Marker | Run | N | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Frame Processing` | Before | 899 | 9.14 | 7.63 | 16.66 | 20.12 |
| `Frame Processing` | After | 899 | 7.61 | 6.53 | 13.25 | 18.60 |
| `Process CPU Frame` | Before | 899 | 2.61 | 1.19 | 7.06 | 10.78 |
| `Process CPU Frame` | After | 900 | 1.85 | 1.33 | 4.25 | 7.07 |
| `Process Hand Frame` | Before | 445 | 1.97 | 2.77 | 3.63 | 4.78 |
| `Process Hand Frame` | After | 460 | 0.32 | 0.38 | 0.65 | 0.79 |
| `MediaPipe Hand Inference` | Before | 232 | 78.22 | 77.28 | 122.33 | 126.76 |
| `MediaPipe Hand Inference` | After | 232 | 87.56 | 83.03 | 119.92 | 129.38 |

Scope of these markers:

- `Frame Processing` measures the synchronous frame callback from buffer receipt
  until it returns. It excludes upstream inference and completion of background
  MediaPipe, browser, and recording work. It is not camera-to-browser latency.
- `Process CPU Frame` wraps the command/overlay-processing call inside that
  callback, rather than the entire CPU-frame method.
- `Process Hand Frame` wraps the hand detector's `process()` call in the stream's
  camera-callback thread. It prepares/enqueues eligible input and returns the
  latest available result; it does not wait for recognition. Some calls skip
  preparation because of frame-rate or ROI gating. This is a thread within the
  main stream process, not necessarily its main thread.
- `MediaPipe Hand Inference` measures the synchronous recognition call in the
  separate worker, including internal waits. It is not pure CPU execution time.

The observed frame-callback p50/p95/p99 decreased by **14.4% / 20.5% / 7.6%**.
The hand-submission marker also became shorter, but its eligible/skipped call
mix differs between captures. Inference itself shows **no consistent gain**:
p50 and p99 rose while p95 fell. Both captures contain 232 completed recognition
calls; equal counts do not establish identical image or hand-detection workloads.

## Preparation stages

These markers describe different work before and after the conversion, so they
are listed separately. Values are milliseconds.

| Run | Marker | N | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Before | `RGB to BGR Conversion` | 899 | 5.13 | 4.67 | 7.71 | 10.70 |
| After | `RGB Frame Copy` | 899 | 4.34 | 3.90 | 6.83 | 10.33 |
| Before | `Hand Frame Encoding` | 286 | 2.89 | 2.77 | 3.61 | 4.61 |
| After | `Hand Frame Snapshot` | 318 | 0.33 | 0.31 | 0.49 | 0.70 |
| After | `Hand Frame Prepare` | 232 | 0.27 | 0.19 | 0.74 | 1.26 |
| After | `Hand RGB Image` | 232 | 0.25 | 0.24 | 0.37 | 0.57 |

The old conversion marker includes the host copy as well as channel conversion.
The new copy marker omits the channel swap. `Hand Frame Snapshot` is the crop/copy
in the camera callback; `Hand Frame Prepare` is resize/shared-memory copy in the
sender thread; `Hand RGB Image` constructs the worker's MediaPipe image. These
stages execute on different threads/processes, and their percentiles must not be
added to estimate end-to-end latency.

## Average utilization

| Measurement | Before | After | Scope |
| --- | ---: | ---: | --- |
| Main-process CPU utilization across six cores | 21.77% | 19.68% | Main stream process only. |

### CPU calculation

CPU time is the sum of measured execution time across all captured threads of the
**main stream process only**. Six-core utilization divides this by elapsed capture
time and six cores. On the single-core scale, 100% means one fully occupied core.

| Run | Scheduled CPU time | Average busy cores | Six-core utilization | Single-core scale |
| --- | ---: | ---: | ---: | ---: |
| Before | 39.187 s | 1.306 | 21.77% | 130.63% |
| After | 35.430 s | 1.181 | 19.68% | 118.10% |

Main-process utilization fell by **2.09 percentage points**, approximately
**9.6% relative**. This does not establish a reduction in total application or
whole-system CPU usage: both reports' scheduling exports contain only the main
stream PID. MediaPipe, HTTP encoding, recording, and other system processes are
absent from this calculation. Changes to work in those processes, particularly
the recording encoder, need separate measurement.

## Evidence and calculation

The original `.nsys-rep` files remain local and are not included in this notes
repository. The derived measurements are preserved here:

- [Latency values and sample counts](../data/2026-09-28-rgb/latency.json)
- [CPU totals, per-core shares, and scheduling checks](../data/2026-09-28-rgb/cpu.json)

Latency comes from `NVTX_EVENTS` and `StringIds`, grouped by process/domain and
marker. Duration is `(end - start) / 1e6` ms. Percentiles use NumPy's linear
interpolation method. Only completed ranges wholly inside the capture window
are included. One invalid/out-of-window range each was excluded from the old
`Frame Processing` and `RGB to BGR Conversion` markers. Nested ranges overlap;
these measurements cannot be summed to recover total frame latency.

CPU time comes from matching `SCHED_EVENTS` schedule-in/out pairs by CPU and
thread, clipped to the `RUN_DURATION_MS` capture window. `CpuCores` in
`TARGET_INFO_SYSTEM_ENV` is six for both runs. All scheduling events paired
without unmatched boundaries or thread mismatches. The computation is:

```text
average busy cores = summed scheduled CPU seconds / capture seconds
six-core CPU utilization (%) = average busy cores / 6 * 100
```

The result is an observation from one run per version. Different scene activity,
ROI sizes, background load, and capture overhead can affect the comparison.
A controlled replay with scheduling coverage for every worker is needed to
attribute total-system performance changes.
