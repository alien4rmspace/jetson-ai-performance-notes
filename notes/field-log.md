# Historical performance field log

This is a sanitized copy of the original experiment log from May 2026. It preserves the measurements, commands, failures, and changing preferences as recorded at the time. Commands may refer to old scripts or a previous JetPack/DeepStream version; they are historical evidence, not current setup instructions. Local account paths and LAN addresses have been replaced with `$PROJECT_ROOT` and documentation addresses.

---

# Performance Gains

Date: 2026-05-04

## Hardware And Runtime

- Board: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- SOC family: `tegra234`
- Power mode: `MAXN_SUPER`
- Clock lock command: `sudo jetson_clocks`
- Verified locked clocks:
  - CPU: `1728 MHz`
  - GPU: `1020 MHz`
  - EMC: `2133 MHz`

## YOLO Stream Baseline

- Script: `web_yolo_stream.py`
- Camera path: Argus/GStreamer CSI camera
- Model: `yolov8n.pt`
- Image size: `imgsz=320`
- Stream endpoint: `http://127.0.0.1:8000`
- LAN endpoint observed: `http://192.0.2.10:8000`

## Measured Improvements

| Configuration | Model | Inference Device | Observed FPS | Relative To CPU |
| --- | --- | --- | ---: | ---: |
| CPU inference | `yolov8n.pt` | CPU | `8.5` | `1.0x` |
| PyTorch CUDA inference | `yolov8n.pt` | GPU `0` / Orin | `23.7` | `2.8x` |
| TensorRT FP16 inference | `yolov8n.engine` | TensorRT / GPU | `30.0` | `3.5x` |
| TensorRT FP16 plus async MediaPipe Gesture Recognizer | `yolov8n.engine` | TensorRT / GPU plus MediaPipe worker | `30.0` | `3.5x` |
| TensorRT object YOLO plus TensorRT YOLO pose plus async hand gestures | `yolov8n.engine` + `yolov8n-pose.engine` | TensorRT / GPU plus MediaPipe worker | `30.3` | `3.6x` |
| Direct TensorRT detector path plus throttled web encode, YOLO `640`, pose `480` | `yolov8n.engine` + `yolov8n-pose.engine` | TensorRT / GPU plus MediaPipe worker | `29.7-30.2` | `3.5x` |

TensorRT is about `1.27x` faster than PyTorch CUDA in the observed stream path.

## Changes Made

- Switched Jetson from `15W` mode to `MAXN_SUPER`.
- Ran `sudo jetson_clocks` to lock clocks.
- Changed both YOLO scripts to default to GPU inference:
  - `run_csi_yolo.py`: `--device-infer 0`
  - `web_yolo_stream.py`: `--device-infer 0`
- Added explicit `task="detect"` when loading YOLO models so TensorRT engine startup does not need task guessing.
- Exported TensorRT FP16 engine:
- Added optional MediaPipe Gesture Recognizer with an isolated `.venv-mediapipe` worker process so MediaPipe's `protobuf<5` dependency does not break MAVSDK in the main `.venv`.
- Made hand detection asynchronous; the YOLO stream keeps running while hand landmarks/gesture labels update in the background.
- Downloaded the trained Gesture Recognizer model: `gesture_recognizer.task`.
- Cropped MediaPipe input to the largest YOLO `person` box by default, with landmarks mapped back to full-frame coordinates for overlay.
- Added landmark-based finger counting alongside the trained gesture label.
- Exported `yolov8n-pose.pt` to TensorRT as `yolov8n-pose.engine`.
- Added optional YOLO pose overlay and body gesture labels.
- Rebuilt `yolov8n-pose.engine` for `pose-imgsz=480`.
- Added a stale-frame watchdog and `src/drone_ai/apps/vision_supervisor.py`; camera/inference stalls now restart the vision process instead of serving a frozen frame indefinitely.
- Added a direct TensorRT detector path for `.engine` detection models, bypassing per-frame `YOLO.predict()` overhead while preserving NMS/result handling.
- Throttled web JPEG encoding to `--stream-fps` so inference frames do not all pay the browser streaming cost.

```bash
cd $PROJECT_ROOT
.venv/bin/yolo export model=yolov8n.pt format=engine imgsz=320 half=True device=0
```

Generated files:

- `yolov8n.engine`
- `yolov8n.onnx`

## Current Preferred Run Command

```bash
cd $PROJECT_ROOT
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000
```

Current verified TensorRT stream state:

- Process observed: PID `5419`
- `/stats`: `FPS: 30.0 | detections: 2 | imgsz: 320`

Current verified TensorRT plus MediaPipe Gesture Recognizer state:

- Process observed: PID `8083`
- Worker observed: PID `8093`
- Command: `.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --hands --hands-model gesture_recognizer.task`
- `/stats`: `FPS: 30.0 | detections: 2 | imgsz: 320 | hands: 0 | gestures: none`

Current verified TensorRT plus Gesture Recognizer with YOLO person crop:

- Process observed: PID `8950`
- Worker observed: PID `8960`
- Command: `.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --hands --hands-model gesture_recognizer.task`
- `/stats`: `FPS: 29.9 | detections: 1 | imgsz: 320 | hands: 1 | gestures: Victory:0.88/fingers:3`

Current verified object YOLO plus YOLO pose plus hand gestures state:

- Process observed: PID `13806`
- Worker observed: PID `13817`
- Command: `.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --pose --pose-model yolov8n-pose.engine`
- `/stats`: `FPS: 27.4 | detections: 3 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: standing`
- Pipeline: full-frame object YOLO at `640`, then crop-based YOLO pose at `320` and crop-based MediaPipe hand gestures.
- Resource sample at `imgsz=640` with crop-based pose: RAM about `2.9 GB / 7.6 GB`, GPU GR3D about `7-37%`, power about `9.3 W`, CPU/GPU temperature about `55 C`.

Current verified optimized stream state:

- Supervisor observed: `src/drone_ai/apps/vision_supervisor.py`
- Stream command: `.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --pose --pose-model yolov8n-pose.engine --pose-imgsz 480 --exit-on-stall`
- `/stats`: `FPS: 29.7-30.2 | detections: 3-4 | imgsz: 640 | hands: active | poses: 1 | body: active`
- Pipeline: full-frame direct TensorRT object YOLO at `640`, crop-based TensorRT YOLO pose at `480`, crop-based async MediaPipe hand gestures, web JPEG encode throttled to `--stream-fps`.
- Resource sample: RAM about `2.9 GB / 7.6 GB`, available RAM about `4.4 GB`, GPU GR3D about `0-29%`, power about `9.4-9.7 W`, CPU/GPU temperature about `54-55 C`.
- Process sample: stream process about `94% CPU` and `959 MB RSS`; MediaPipe worker about `68% CPU` and `209 MB RSS`; `nvargus-daemon` about `14% CPU` and `152 MB RSS`.

## Further Efficiency Options

- Lower `--imgsz` to `256` if more speed is needed.
- Lower stream rendering cost with `--max-stream-width 640 --jpeg-quality 65 --stream-fps 8`.
- Add an `--infer-every` option to avoid running detection on every frame.
- Skip `result.plot()` when detections are needed for control logic but not for visual display.

## Resource Usage Log

### 2026-05-04 02:00 local

Pipeline:

```bash
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --hands --hands-model gesture_recognizer.task --pose --pose-model yolov8n-pose.engine
```

Live stats:

```text
FPS: 29.3 | detections: 1 | imgsz: 320 | hands: 0 | gestures: none | poses: 1 | body: standing
```

System:

```text
RAM: 2.4 GiB / 7.4 GiB used
Available RAM: 4.8 GiB
Swap: about 1 MiB / 3.7 GiB
Load average: 2.00, 2.40, 2.50
Power: about 8.4-8.6 W
CPU/GPU temperature: about 54 C
```

Main process usage:

```text
web_yolo_stream.py:        about 114% CPU, 942 MB RAM
mediapipe_hands_worker.py: about 48% CPU, 224 MB RAM
nvargus-daemon:            about 15% CPU, 159 MB RAM
```

Jetson tegrastats sample:

```text
CPU: one core spikes about 84-94%, other cores about 13-30%
GPU GR3D: about 1-12%
GPU clock: about 1012-1013 MHz
EMC: about 6-7% @ 2133 MHz
VIC: about 11-39%
```

### 2026-05-04 03:05 local

Pipeline:

```bash
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --pose --pose-model yolov8s-pose.engine --pose-imgsz 480 --exit-on-stall
```

Live stats:

```text
FPS: 29.9 | detections: 1 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: standing
```

System:

```text
RAM: 2.5 GiB / 7.4 GiB used
Available RAM: 4.7 GiB
Swap: 15 MiB / 3.7 GiB
Power: about 9.6-9.9 W
CPU/GPU temperature: about 55-56 C
```

Main process usage:

```text
web_yolo_stream.py:        about 89% CPU, 965 MB RAM
mediapipe_hands_worker.py: about 65% CPU, 204 MB RAM
nvargus-daemon:            about 16% CPU, 160 MB RAM
src/drone_ai/apps/vision_supervisor.py:      idle CPU, 10 MB RAM
```

Jetson tegrastats sample:

```text
CPU: about 17-54% per core
GPU GR3D: about 10-38%
GPU clock: about 1004-1015 MHz
EMC: about 13% @ 2133 MHz
VIC: about 12-32%
```

### 2026-05-04 16:00 local

Pipeline:

```bash
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --hands-idle-fps 10 --hands-detection-conf 0.35 --hands-tracking-conf 0.35 --hands-gesture-conf 0.40 --pose --pose-model yolov8s-pose.engine --pose-imgsz 480 --pose-debug --stream-fps 8 --max-stream-width 640 --jpeg-quality 65 --exit-on-stall
```

Live stats:

```text
FPS: 24.8 | detections: 2 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: standing
```

Main process usage:

```text
web_yolo_stream.py:        about 96% CPU, 1003 MB RSS
mediapipe_hands_worker.py: about 49% CPU, 222 MB RSS
nvargus-daemon:            about 20% CPU, 176 MB RSS
```

Current command-layer performance notes:

- Browser MJPEG cost was reduced with `--stream-fps 8`, `--max-stream-width 640`, and `--jpeg-quality 65`.
- `/stats` and `/command` request logging was suppressed to reduce log spam.
- Hand detection felt weak at idle with lower sampling, so idle hand sampling was raised to `10 FPS`.
- MediaPipe confidence thresholds were lowered to improve hand acquisition:
  - `--hands-detection-conf 0.35`
  - `--hands-tracking-conf 0.35`
  - `--hands-gesture-conf 0.40`
- Tradeoff: MediaPipe worker CPU rises when hands are active or sampled more often.
- Pose debug remains enabled for command tuning; disabling `--pose-debug` is still a likely easy efficiency win after tuning.

### 2026-05-04 16:29 local

GPU clocks were observed low before re-locking:

```text
GR3D_FREQ: about 29-73% @[305]
Stream: about 24.9 FPS
Power mode: MAXN_SUPER
```

After running:

```bash
sudo jetson_clocks
```

Verified clocks and stream:

```text
CPU: 1728 MHz
GPU: about 1004-1016 MHz
EMC: 3199 MHz
Stream: FPS: 30.2 | detections: 3 | imgsz: 640 | hands: 1 | poses: 1
```

Performance conclusion:

- GPU utilization below `100%` does not mean GPU clock does not matter.
- TensorRT object/pose inference happens in short GPU bursts, with CPU/camera/MediaPipe/JPEG work between bursts.
- A low GPU clock makes each burst slower, which increases frame latency and can reduce total pipeline FPS even if average GR3D utilization is moderate.
- Locking clocks restored the expected `~30 FPS` stream behavior.

### 2026-05-04 end-of-session log

Latest active stream configuration before stopping:

```bash
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --hands-idle-fps 10 --hands-detection-conf 0.35 --hands-tracking-conf 0.35 --hands-gesture-conf 0.40 --pose --pose-model yolov8s-pose.engine --pose-imgsz 480 --pose-debug --stream-fps 8 --max-stream-width 640 --jpeg-quality 65 --exit-on-stall
```

Observed sample after raising hands idle FPS to `10`:

```text
FPS: 30.0 | detections: 3 | imgsz: 640 | hands: 1 | gestures: None:0.87/fingers:4 | poses: 2 | body: pose_unknown, pose_unknown
```

Process sample:

```text
mediapipe_hands_worker.py: about 77% CPU, 210 MB RSS
web_yolo_stream.py:        about 77% CPU, 990 MB RSS
nvargus-daemon:            about 12% CPU, 165 MB RSS
```

End state:

```text
src/drone_ai/apps/vision_supervisor.py, web_yolo_stream.py, and mediapipe_hands_worker.py stopped
http://127.0.0.1:8000/stats: connection refused
```

Notes:

- `--hands-idle-fps 10` improves hand responsiveness compared with lower idle sampling.
- Higher hand sampling increases MediaPipe CPU use, but with Jetson clocks locked the stream still held about `30 FPS` in the observed sample.
- Pose debug remains enabled for gesture tuning; remove `--pose-debug` later for a cleaner overlay and possible small efficiency gain.

### 2026-05-06 YOLO26 pose upgrade

Changed pose inference from `yolov8s-pose.engine` to `yolo26s-pose.engine` in `src/drone_ai/apps/vision_supervisor.py`.

Export command:

```bash
.venv/bin/yolo export model=yolo26s-pose.pt format=engine imgsz=480 half=True device=0
```

Export result:

```text
YOLO26s-pose summary: 132 layers, 10,359,750 parameters, 23.9 GFLOPs
TensorRT FP16 engine: yolo26s-pose.engine, 23.9 MB
Engine generation time: about 487 seconds
```

Performance note:

- Runtime FPS has not been re-measured yet after switching the supervisor to YOLO26s pose.
- This model is larger than the previous YOLOv8 small pose engine, so pose quality may improve but pose inference cost may increase.

### 2026-05-06 live resource and GPU sample

Live stream stats after switching pose to YOLO26s:

```text
FPS: 30.0 | detections: 2 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: standing
```

Process sample:

```text
web_yolo_stream.py:        about 72.7% CPU, 1001 MB RSS
mediapipe_hands_worker.py: about 60.7% CPU, 216 MB RSS
nvargus-daemon:            about 13.8% CPU, 167 MB RSS
```

System sample:

```text
RAM: 2.7 GiB used / 7.4 GiB total, 4.5 GiB available
Swap: 0 B used
Load average: 2.51, 1.31, 0.73
```

Jetson sample:

```text
CPU cores: about 16-35% at 1728 MHz
GR3D_FREQ: 1-25%@[1012-1015]
EMC_FREQ: 8-9% at 3199 MHz
Temps: about 59 C CPU/GPU
Power: about 10.6-10.9 W
```

Interpretation:

- `GR3D_FREQ` is the Jetson 3D/GPU engine report.
- `25%@[1015]` means roughly `25%` GPU utilization at about `1015 MHz`.
- GPU clock is high/locked; GPU utilization is low to moderate.
- Low GPU utilization is acceptable here because TensorRT YOLO runs in short bursts, then waits on CPU/camera/MediaPipe/JPEG/HTTP work.
- Current performance bottleneck remains CPU-side work rather than GPU capacity.

### 2026-05-06 MediaPipe VIDEO mode hand tracking sample

Changed the MediaPipe hand worker from per-frame `IMAGE` mode to `VIDEO` mode.

Implementation:

```text
running_mode=vision.RunningMode.VIDEO
recognizer.recognize_for_video(mp_image, timestamp_ms)
```

The worker now sends strictly increasing monotonic timestamps to MediaPipe so the Tasks API can use temporal hand tracking across sequential frames.

Rationale:

- `IMAGE` mode treats each hand input as unrelated.
- `VIDEO` mode fits the existing synchronous stdin/stdout worker and can use frame-to-frame tracking.
- `LIVE_STREAM` mode is callback-based and was not needed for the current architecture.

Live stream stats after restart:

```text
FPS: 29.8 | detections: 1 | imgsz: 640 | hands: 1 | gestures: None:0.62/fingers:0 | poses: 1 | body: pose_unknown
```

Process sample:

```text
web_yolo_stream.py:        about 71.1% CPU, 1005 MB RSS
mediapipe_hands_worker.py: about 44.7% CPU, 224 MB RSS
src/drone_ai/apps/vision_supervisor.py:      idle, 10 MB RSS
```

Jetson sample:

```text
RAM 2845/7620MB, SWAP 0/3810MB
CPU cores: about 12-43% at 1728 MHz
EMC_FREQ: 9%@3199
GR3D_FREQ: 37%@[1011]
VIC: 41%@115
Temps: about 59-61 C
Power: VDD_IN 11078mW, VDD_CPU_GPU_CV 4126mW, VDD_SOC 3306mW
```

Notes:

- The YOLO object detector still crops the detected `person`, not the hand.
- MediaPipe receives the padded person crop and performs hand detection/landmarking inside that crop.
- Current object detector does not provide a `hand` class.
- Future tests for hand landmark quality should try larger person crops or higher hand input width before adding a dedicated YOLO hand detector.

### 2026-05-06 gesture tuning resource sample before shutdown

Final running command included:

```text
--hands-idle-fps 10 --hands-fps 20 --pose --pose-debug --max-stream-width 960 --jpeg-quality 65
```

Latest observed stream stats before shutdown:

```text
FPS: 29.8 | detections: 2 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: pose_unknown
```

Representative process sample:

```text
web_yolo_stream.py:        about 75.7% CPU, 999 MB RSS
mediapipe_hands_worker.py: about 49.1% CPU, 215 MB RSS
nvargus-daemon:            about 13.8% CPU, 165 MB RSS
src/drone_ai/apps/vision_supervisor.py:      idle, 10 MB RSS
```

System sample:

```text
RAM: 2.7 GiB used / 7.4 GiB total, 4.5 GiB available
Swap: 0 B used / 3.7 GiB total
Load average: 1.81, 1.91, 1.60
```

Jetson sample:

```text
CPU cores: about 17-40% at 1728 MHz
GR3D_FREQ: 21%@[1011]
EMC_FREQ: 8%@3199
Temps: CPU about 60.6 C, GPU about 61.1 C
Power: VDD_IN about 11.0 W
```

Power/current note:

```text
VDD_IN sensor: 4.928 V, 2.19 A, about 10.5 W
Estimated 19 V adapter draw: about 0.55 A ideal, roughly 0.6 A with regulator losses
```

End state:

```text
src/drone_ai/apps/vision_supervisor.py, web_yolo_stream.py, and mediapipe_hands_worker.py stopped
http://127.0.0.1:8000/stats: connection refused
```

Notes:

- Raising active hand inference to `20 FPS` did not prevent the stream from holding about `30 FPS` in the observed sample.
- MediaPipe CPU usage varies with visible hands; the sample above was with hands idle/not detected.
- Pose debug remains enabled and costs some overlay work; disable later for cleaner display and a small efficiency gain.

### 2026-05-07 idle storage and resource baseline

Storage layout:

```text
$PROJECT_ROOT is on /dev/nvme0n1p1, ext4, mounted as /
NVMe model: Fanxiang S500Pro 256GB
Root filesystem: 234G total, 28G used, 194G available, 13% used
SD card mmcblk0 is present; only /boot/efi is mounted from mmcblk0p10
```

Memory sample while stream was stopped:

```text
RAM: 1.5 GiB used / 7.4 GiB total, 5.7 GiB available
Swap: 0 B used / 3.7 GiB total
```

Jetson idle sample while stream was stopped:

```text
CPU cores: about 0-2% at 1728 MHz
GR3D_FREQ: 0%@[1017]
EMC_FREQ: 0%@3199
Temps: CPU about 53.3 C, GPU about 54.2 C
Power: VDD_IN about 6.96 W
```

Notes:

- The current project, models, virtual environments, and logs are on NVMe SSD storage, not the SD card.
- The stream was not running for this sample, so this is an idle/baseline reading.

### 2026-05-07 live stream resource sample

Pipeline:

```bash
.venv/bin/python web_yolo_stream.py --model yolov8n.engine --port 8000 --imgsz 640 --hands --hands-model gesture_recognizer.task --hands-idle-fps 10 --hands-fps 20 --hands-detection-conf 0.35 --hands-tracking-conf 0.35 --hands-gesture-conf 0.40 --pose --pose-model yolo26s-pose.engine --pose-imgsz 480 --pose-debug --stream-fps 8 --max-stream-width 960 --jpeg-quality 65 --exit-on-stall
```

Live stats:

```text
FPS: 30.0 | detections: 1 | imgsz: 640 | hands: 0 | gestures: none | poses: 1 | body: left_hand_up
```

Process usage:

```text
web_yolo_stream.py:        about 79.0% CPU, 1000 MB RSS
mediapipe_hands_worker.py: about 59.8% CPU, 212 MB RSS
nvargus-daemon:            about 13.9% CPU, 173 MB RSS
src/drone_ai/apps/vision_supervisor.py:      about 0.0% CPU, 10 MB RSS
```

Memory:

```text
RAM: 1.8 GiB used / 7.4 GiB total, 5.4 GiB available
Swap: 0 B used / 3.7 GiB total
```

Jetson `tegrastats`:

```text
CPU cores: about 13-41% at 1728 MHz
GR3D_FREQ: 0-25%@[1012-1013]
EMC_FREQ: 8%@3199
VIC: 10-25%@115
Temps: CPU about 58.0 C, GPU about 58.4 C
Power: VDD_IN about 10.8-11.1 W
```

Notes:

- Stream was holding about `30 FPS` with TensorRT object detection, YOLO26 pose, async MediaPipe hands, `--pose-debug`, and browser MJPEG capped at `960px` / `8 FPS`.
- Sample had no detected hands, but the MediaPipe worker still used about `60% CPU`.

### 2026-05-07 DeepStream YOLO26S RTSP object stream

Pipeline:

```text
nvarguscamerasrc -> video/x-raw(memory:NVMM) -> nvstreammux -> nvinfer YOLO26S -> nvvideoconvert -> nvdsosd -> x264enc -> RTSP
```

Command:

```bash
python3 src/drone_ai/apps/vision_stream.py --sink rtsp --rtsp-host 192.0.2.10
```

RTSP URL:

```text
rtsp://192.0.2.10:8554/ds
```

Live stats:

```text
FPS: 30.0 | detections: 1-2 | persons: 1-2 | sink: rtsp
```

Process/resource sample:

```text
src/drone_ai/apps/vision_stream.py: about 58% CPU, 363 MB RSS
nvargus-daemon:              about 6.6% CPU, 221 MB RSS
RAM:                         about 2.3 GiB / 7.4 GiB used
GR3D_FREQ:                   about 23%@[1013]
VIC:                         about 22%@115
Temps:                       CPU about 55.1 C, GPU about 56.2 C
Power:                       VDD_IN about 10.7 W
```

Notes:

- `nvv4l2h264enc` is not available on this Orin Nano setup, so RTSP uses CPU `x264enc` through `--encoder auto`.
- The DeepStream camera and YOLO26S inference path stays NVMM/DeepStream; only the H264 network output pays the software encode cost.
- Use `--sink fake` for headless inference benchmarking or `--sink egl` for local display without RTSP encode.

### 2026-05-07 DeepStream YOLO26S HTTP browser stream

Pipeline:

```text
nvarguscamerasrc -> video/x-raw(memory:NVMM) -> nvstreammux -> nvinfer YOLO26S -> nvvideoconvert -> nvdsosd -> nvvideoconvert -> nvjpegenc -> HTTP MJPEG
```

Command:

```bash
python3 src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080
```

Browser URL:

```text
http://192.0.2.10:8080
```

Live stats:

```text
FPS: 30.0 | detections: 2 | persons: 1 | sink: http
```

Resource sample:

```text
src/drone_ai/apps/vision_stream.py: about 26.6% CPU, 364 MB RSS
nvargus-daemon:              about 7.4% CPU, 229 MB RSS
RAM:                         about 2.3 GiB / 7.4 GiB used
GR3D_FREQ:                   about 43%@[1016]
NVJPG1:                      about 6%@[499]
VIC:                         about 25%@115
Temps:                       CPU about 56.3 C, GPU about 57.7 C
Power:                       VDD_IN about 10.4 W
```

Notes:

- This is the current low-latency browser path.
- Browser output is multipart MJPEG; camera/inference remain in DeepStream/NVMM and JPEG encoding uses NVIDIA `nvjpegenc`.

### 2026-05-07 DeepStream browser stream with legacy commands

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --hands --pose --pose-model yolo26s-pose.engine --pose-imgsz 480 --hands-idle-fps 10 --hands-fps 20
```

Live stats:

```text
FPS: 30.0 | detections: 2 | persons: 1 | sink: http | hands: 0 | gestures: none | poses: 1 | body: standing
/command: No command
```

Process usage:

```text
src/drone_ai/apps/vision_stream.py: about 65% CPU, 1104 MB RSS
mediapipe_hands_worker.py:  about 39% CPU, 210 MB RSS
nvargus-daemon:             about 8% CPU, 226 MB RSS
```

Notes:

- The browser layout now matches the original `web_yolo_stream.py` layout with the command text field above the stream.
- `/command` is available and polled every `500 ms`.
- Legacy hand and pose command classification are enabled from a side branch that decodes the DeepStream JPEG frame. The main camera/object path remains DeepStream/NVMM with YOLO26S object detection.

### 2026-05-07 DeepStream browser stream with TensorRT hands

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --hands --hand-backend trt --pose --pose-model yolo26s-pose.engine --pose-imgsz 480 --hands-idle-fps 10 --hands-fps 20
```

Live stats sample:

```text
FPS: 29.9 | detections: 2 | persons: 2 | sink: http | hands(trt): 2 | gestures: Closed_Fist:0.70/fingers:0, Thumb_Up:0.65/fingers:0 | poses: 1 | body: standing
/command: No command
```

Process usage:

```text
src/drone_ai/apps/vision_stream.py: about 81.5% CPU, 1117704 KB RSS
mediapipe_hands_worker.py:  not running
web_yolo_stream.py:         not running
```

System and Jetson sample:

```text
RAM:       about 2589 MB / 7620 MB used
Swap:      0 MB / 3810 MB
CPU:       one core about 48-67% during samples, other cores mostly about 5-36%
GPU GR3D:  about 17-58% @ ~1001-1016 MHz
VIC:       about 11-38%
NVJPG:     about 0-7%
EMC:       about 12% @ 3199 MHz
Temps:     CPU about 58.8-59.2 C, GPU about 59.8-60.3 C
Power:     VDD_IN about 11.8-12.0 W
```

Comparison to old web stream plus MediaPipe:

```text
Old web_yolo_stream.py + MediaPipe: main about 79% CPU / 1000 MB RSS plus MediaPipe about 60% CPU / 212 MB RSS, power about 10.8-11.1 W.
Current DeepStream + TensorRT hands: main about 81.5% CPU / 1118 MB RSS, no MediaPipe worker, power about 11.8-12.0 W.
```

Notes:

- This is the current best command-capable browser stream.
- The hand backend now uses TensorRT palm detection and TensorRT hand landmarks:
  - `models/qualcomm_mediapipe_hand/HandDetector.fp16.engine`
  - `models/qualcomm_mediapipe_hand/HandLandmarkDetector.fp16.engine`
  - `models/qualcomm_mediapipe_hand/anchors_palm.npy`
- CPU is lower overall than the old web stream because the separate MediaPipe worker is gone.
- Power is slightly higher than the old web stream because more work is now running through TensorRT/GPU, and YOLO pose plus overlay/JPEG drawing are still active.
- Memory is slightly lower overall than old web stream plus MediaPipe worker, but the main process is still large because DeepStream, Torch/Ultralytics pose, and TensorRT engines are loaded.

### 2026-05-08 DeepStream browser stream with CPU MediaPipe hands

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --hands --hand-backend mediapipe --pose --pose-model yolo26s-pose.engine --pose-imgsz 480 --hands-idle-fps 10 --hands-fps 20
```

Live stats sample:

```text
FPS: 30.1 | detections: 2 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: standing
/command: No command
```

Process usage:

```text
src/drone_ai/apps/vision_stream.py: about 83.2% CPU, 1113880 KB RSS
mediapipe_hands_worker.py:  about 51.1% CPU, 202724 KB RSS
web_yolo_stream.py:         not running
```

System and Jetson sample:

```text
RAM:       about 3189-3190 MB / 7620 MB used
Swap:      about 1 MB / 3810 MB
CPU:       about 22-24% average per core across 6 cores
GPU GR3D:  about 24-48%, roughly 33% average
VIC:       about 16-46%
NVJPG:     about 0-3%
EMC:       about 11% @ 3199 MHz
Temps:     CPU about 59.1 C, GPU about 59.8-60.2 C
Power:     VDD_IN about 11.6-11.75 W
```

Notes:

- This is the current preferred live mode because closed-fist gesture recognition is better than the TensorRT landmark-only hand path.
- DeepStream still handles the CSI camera, YOLO26S object detection, overlay, JPEG encode, and browser stream.
- MediaPipe runs only as the hand gesture worker from `.venv-mediapipe`; the old `web_yolo_stream.py` is not running.
- Compared with the TensorRT hand path, this mode uses more CPU because the separate MediaPipe worker is back, but gesture accuracy for closed fists is currently better.

### 2026-05-08 HTTP frame copy optimization

Change:

```text
Old HTTP path: GPU/NVMM frame -> nvjpegenc -> CPU JPEG -> cv2.imdecode -> CPU BGR frame -> cv2.imencode -> browser
New HTTP path: GPU/NVMM frame -> nvvideoconvert raw BGRx appsink -> CPU BGR frame -> cv2.imencode -> browser
```

Notes:

- CPU MediaPipe hands still require a CPU frame, so one GPU-to-CPU transfer remains.
- The avoidable JPEG decode round trip was removed.
- This was true before the later DeepStream pose SGIE change below; current pose command control now uses DeepStream tensor metadata.

Post-change live sample:

```text
FPS: 30.0 | detections: 3 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: standing
src/drone_ai/apps/vision_stream.py: about 82.6% CPU, 1103060 KB RSS
mediapipe_hands_worker.py: about 40.3% CPU, 197916 KB RSS
RAM: about 3177-3184 MB / 7620 MB used
GPU GR3D: about 0-40%
VIC: about 14-48%
NVJPG/NVJPG1: off
Temps: CPU about 57.2-58.0 C, GPU about 58.2-58.7 C
Power: VDD_IN about 11.7-12.2 W
```

### 2026-05-08 DeepStream YOLO26S pose SGIE

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --hands --hand-backend mediapipe --pose --pose-backend deepstream --hands-idle-fps 10 --hands-fps 20
```

Change:

```text
Old pose path: CPU BGR frame -> legacy Ultralytics PoseDetector -> TensorRT pose engine -> CPU keypoints
New pose path: DeepStream person object -> yolo26s-pose SGIE nvinfer -> tensor metadata -> CPU keypoints only
```

Live validation:

```text
FPS: 30.0 | detections: 2 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: standing
/command: No command
```

Notes:

- `deepstream/config_yolo26s_pose_sgie.txt` runs `yolo26s-pose.onnx` as a secondary GIE on person boxes from the object detector.
- The generated local engine is `deepstream/yolo26s_pose_sgie_b1_fp16.engine`.
- The active object and pose SGIE engines were rebuilt with system TensorRT 10.3 via `trtexec --builderOptimizationLevel=3 --fp16 --memPoolSize=workspace:1024 --skipInference`.
- Pose commands now come from DeepStream tensor metadata, not the legacy Ultralytics pose path.

### 2026-05-12 DeepStream HTTP browser publish cap

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --hands --hand-backend mediapipe --pose --pose-backend deepstream --hands-idle-fps 10 --hands-fps 20
```

Change:

```text
Added --http-stream-fps, default 20.0.
DeepStream camera/inference and command/gesture processing keep their existing cadence.
Browser overlay drawing and cv2.imencode(".jpg", ...) are skipped on non-publish frames.
Person-lock clothing histogram compare/update remains throttled separately at --person-lock-appearance-fps 15.0.
```

Live validation after restart:

```text
FPS: 30.0 | detections: 1 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: left_hand_up
```

Process usage sample:

```text
src/drone_ai/apps/vision_stream.py: about 79.9% CPU, 859 MB RSS
mediapipe_hands_worker.py:  about 70.2% CPU, 209 MB RSS
nvargus-daemon:             about 12.0% CPU, 159 MB RSS
```

Thread CPU sample for `src/drone_ai/apps/vision_stream.py`:

```text
PID 41654: src/drone_ai/apps/vision_stream.py
TID 41709: 38.1% CPU, command "python", hottest native OpenCV/Python image-processing thread
TID 41654: 18.4% CPU, command "python", main GLib/GStreamer polling thread
TID 41713: 10.1% CPU, command "python", DeepStream/inference helper thread
TID 41710: 9.9% CPU, command "python", DeepStream/inference helper thread
TID 41727: 7.3% CPU, command "argus_thread", camera/Argus path
TID 41716: 1.5% CPU, command "consumer_thread", camera/VIC transform path
TID 41720: 1.3% CPU, command "stream_mux:src", nvstreammux source thread
TID 41726: 1.1% CPU, command "EglStrmComm*111", Argus/EGL stream communication
TID 41708: 0.3% CPU, command "cuda-EvtHandlr", CUDA event handler
```

System and Jetson sample:

```text
RAM: 2942 MB / 7620 MB used
Swap: 0 MB / 3810 MB used
CPU clocks: 1728 MHz on all cores
GPU GR3D: about 22%
VIC: about 39%
NVDEC: off
NVJPG/NVJPG1: off
Power VDD_IN: about 13.6 W
Temps: CPU about 60.2 C, GPU/TJ about 61.3 C
```

Comparison notes:

```text
Before 20 FPS browser publish cap, recent main-process samples were around 88-98% CPU.
With 20 FPS browser publish cap, the measured main process sample dropped to about 80% CPU.
The hottest OpenCV/Python thread dropped from roughly 55.5% CPU in a camera-FPS publish sample to roughly 38.1% CPU.
```

PID/thread notes:

```text
A PID is a process, not a CPU core. Each process can have many OS threads.
Linux schedules those threads across available cores, so one PID can use more than one core at once if multiple threads are runnable.
Process CPU percentages are the sum of all its threads: 100% is about one full CPU core, 200% is about two full CPU cores.
The TID/SPID rows above are individual threads inside the src/drone_ai/apps/vision_stream.py process.
```

### 2026-05-12 4K capture with 1080p CPU hand path

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --width 3840 --height 2160 --framerate 30 --cpu-frame-width 1920 --cpu-frame-height 1080 --hands --hands-person-crop --hand-backend mediapipe --pose --pose-backend deepstream --hands-idle-fps 10 --hands-fps 15
```

Change:

```text
CSI camera and DeepStream/YOLO/pose run from a 3840x2160 @ 30 FPS source.
The HTTP/appsink CPU frame is downscaled by GStreamer/VIC to 1920x1080 before CPU mapping.
DeepStream person boxes and pose keypoints are scaled into CPU-frame coordinates before person lock, overlays, hand matching, and command logic.
MediaPipe hands receives the person ROI crop from the 1080p CPU frame via --hands-person-crop.
Active MediaPipe hands cadence is --hands-fps 15; idle cadence remains --hands-idle-fps 10.
```

Live validation:

```text
FPS: 25.6 | detections: 1 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: standing
Browser MJPEG frame size: 1920x1080
Argus camera mode: 3840x2160 @ 29.999999 FPS
```

Process usage sample:

```text
src/drone_ai/apps/vision_supervisor.py:        0.0% CPU, 10 MB RSS
nvargus-daemon:             13.5% CPU, 164 MB RSS
src/drone_ai/apps/vision_stream.py: 113% CPU, 877 MB RSS
mediapipe_hands_worker.py:  67.9% CPU, 201 MB RSS
```

Thread CPU sample for `src/drone_ai/apps/vision_stream.py`:

```text
PID 64742: src/drone_ai/apps/vision_stream.py
TID 64793: 75.9% CPU, command "python", hottest CPU-side stream thread
TID 64800: 9.4% CPU, command "python", waiting on dma_fence_default_wait
TID 64794: 9.4% CPU, command "python"
TID 64742: 3.0% CPU, command "python", main GLib/GStreamer polling thread
TID 64803: 1.8% CPU, command "consumer_thread", camera/VIC path
TID 64798: 1.8% CPU, command "python"
TID 64758: 0.6% CPU, command "python", polling/helper thread
TID 64792: 0.3% CPU, command "cuda-EvtHandlr", CUDA event handler
```

Thread CPU sample for `mediapipe_hands_worker.py`:

```text
PID 64755: mediapipe_hands_worker.py
TID 64842: 10.6% CPU
TID 64841: 10.5% CPU
TID 64839: 10.4% CPU
TID 64843: 10.3% CPU
TID 64840: 10.0% CPU
TID 64844: 10.0% CPU
TID 64755: 3.0% CPU, main worker thread
```

System and Jetson sample:

```text
RAM: 2915-2918 MB / 7620 MB used
Swap: 0 MB / 3810 MB used
CPU clocks: 1728 MHz on all cores
CPU cores: roughly 24-67% depending on core
GPU GR3D: 6-44%
VIC: 64-69%
NVDEC: off
NVJPG/NVJPG1: off
Power VDD_IN: about 13.9-14.1 W
Temps: CPU about 62.3-62.8 C, GPU/TJ about 63.1-63.3 C
```

Notes:

```text
The 4K-to-1080 CPU split recovered the stream from roughly 5-6 FPS at full 4K CPU mapping to roughly 25-28 FPS.
The remaining hot CPU stream thread is still CPU-side frame processing/overlay/JPEG/command work after the 1080p CPU map.
VIC usage is high because it is doing the hardware downscale/format conversion path.
Future resource usage checks should include process usage plus thread/TID breakdowns by default.
```

### 2026-05-12 CPU path timing breakdown

Command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --width 3840 --height 2160 --framerate 30 --cpu-frame-width 1920 --cpu-frame-height 1080 --hands --hands-person-crop --hand-backend mediapipe --pose --pose-backend deepstream --hands-idle-fps 10 --hands-fps 15
```

Live validation:

```text
FPS: 24.7 | detections: 2 | persons: 1 | sink: http | hands(mediapipe): 1 | gestures: None:0.68/fingers:0 | poses: 1 | body: standing
```

Python-side timing sample from `/pose-debug`:

```text
command+overlay+encode: 15.93 ms (49 samples)
jpeg encode: 15.91 ms (32 samples)
person lock: 4.94 ms (49 samples)
command update: 13.55 ms (17 samples)
cpu map+cvtColor: 3.10 ms (49 samples)
overlay draw: 1.20 ms (32 samples)
```

Native stack sample for hottest `src/drone_ai/apps/vision_stream.py` thread:

```text
Hottest TID: 64793, about 75% CPU in the sampled run.
Stack location:
  ioctl
  libnvrm_host1x.so
  NvBufSurfTransformAsync
  libgstnvvideoconvert.so
  GStreamer pad push
  DeepStream libnvdsgst_infer.so
```

Interpretation:

```text
The hottest stream thread was not inside cv2.imencode, cv2.cvtColor, or OpenCV drawing.
It was inside the GStreamer/VIC downscale/format-convert path from 4K/NVMM to the 1080p BGRx appsink frame.
This appears as CPU attributed to src/drone_ai/apps/vision_stream.py because the pipeline runs in that process, but much of the work is driving/waiting on NVIDIA buffer transform hardware.

Among explicit Python/OpenCV CPU sections, JPEG encode is the largest measured cost at about 15.9 ms per encoded browser frame.
MediaPipe command update is also expensive at about 13.6 ms when it runs, but it runs at the hand cadence, not every frame.
Person-lock ROI/histogram work is next at about 4.9 ms per frame.
cvtColor is about 3.1 ms per frame.
Overlay drawing is about 1.2 ms per drawn frame.
```

Current bottleneck ranking:

```text
1. GStreamer nvvideoconvert / NvBufSurfTransformAsync downscale+format-convert path.
2. JPEG encode for browser MJPEG.
3. MediaPipe command update when hand frames are processed.
4. Person lock histogram/ROI work.
5. cvtColor.
6. Overlay drawing.
```

### 2026-05-12 BGR appsink with GPU nvvideoconvert

Purpose:

```text
Test whether the HTTP CPU/appsink path can avoid BGRx and the Python cv2.cvtColor(BGRx -> BGR) step.
```

Negotiation tests:

```bash
gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 num-buffers=90 \
  ! 'video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1' \
  ! nvvideoconvert \
  ! 'video/x-raw,width=1920,height=1080,format=BGR' \
  ! fakesink sync=false
```

Result:

```text
Default/VIC BGR path failed.
Error: RGB/BGR Format transformation is not supported by VIC use GPU instead.
BGR_EXIT: 1
```

Control:

```bash
gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 num-buffers=90 \
  ! 'video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1' \
  ! nvvideoconvert \
  ! 'video/x-raw,width=1920,height=1080,format=BGRx' \
  ! fakesink sync=false
```

Result:

```text
BGRx via default path worked.
BGRX_EXIT: 0
```

GPU compute test:

```bash
gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 num-buffers=90 \
  ! 'video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1' \
  ! nvvideoconvert compute-hw=GPU \
  ! 'video/x-raw,width=1920,height=1080,format=BGR' \
  ! fakesink sync=false
```

Result:

```text
BGR with nvvideoconvert compute-hw=GPU worked.
BGR_GPU_EXIT: 0
```

Implemented live test:

```text
Added --http-frame-format with choices BGRx/BGR.
When --http-frame-format BGR is selected, the HTTP nvvideoconvert uses compute-hw=GPU.
The appsink maps BGR directly and copies it into a CPU-owned NumPy frame instead of doing cv2.cvtColor(BGRx -> BGR).
src/drone_ai/apps/vision_supervisor.py now passes --http-frame-format BGR for the live stream.
```

Live command:

```bash
.venv/bin/python src/drone_ai/apps/vision_stream.py --sink http --http-host 192.0.2.10 --http-port 8080 --width 3840 --height 2160 --framerate 30 --cpu-frame-width 1920 --cpu-frame-height 1080 --http-frame-format BGR --hands --hands-person-crop --hand-backend mediapipe --pose --pose-backend deepstream --hands-idle-fps 10 --hands-fps 15
```

Live validation:

```text
FPS: 30.0 | detections: 2 | persons: 2 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 2 | body: right_hand_up, right_hand_up
```

Timing before:

```text
cpu map+cvtColor: about 3.10 ms per frame
```

Timing after:

```text
cpu map+copy BGR: 1.61 ms (59 samples)
command+overlay+encode: 11.90 ms (59 samples)
jpeg encode: 15.86 ms (29 samples)
person lock: 5.17 ms (60 samples)
command update: 12.90 ms (17 samples)
overlay draw: 0.70 ms (30 samples)
```

Resource sample after BGR/GPU path:

```text
src/drone_ai/apps/vision_stream.py: 112% CPU, 1004 MB RSS
mediapipe_hands_worker.py: 52.7% CPU, 202 MB RSS
nvargus-daemon: 12.1% CPU, 160 MB RSS
GPU GR3D: 21-58%
VIC: 44-68%
Power VDD_IN: about 14.2-14.5 W
Temps: CPU about 60 C, GPU/TJ about 61 C
```

Interpretation:

```text
This is a meaningful improvement for the CPU frame mapping/color step: about 3.10 ms -> 1.61 ms.
Short live test restored reported stream FPS to about 30 FPS.
Default BGR is not reliable on Jetson VIC; direct BGR requires nvvideoconvert compute-hw=GPU.
This likely shifts some work from VIC/default path to GR3D/GPU. Watch stability because the prior 4K path has shown DeepStream CUDA illegal-address crashes under load.
```

### 2026-05-13 Local Qwen LLM GPU Offload

Goal:

```text
Move the local voice-command/conversation LLM from CPU-only llama.cpp execution to Jetson Orin GPU offload.
Keep the voice-command output machine-readable JSON while allowing harmless conversational responses.
```

Implementation:

```text
Installed ninja-build for local CUDA wheel compilation.
Rebuilt llama-cpp-python==0.3.22 with:
  GGML_CUDA=on
  CMAKE_CUDA_ARCHITECTURES=87

Verified llama_supports_gpu_offload() returns True.
Changed voice_commands.py default --n-gpu-layers from 0 to -1, offloading all supported layers by default.
Use --n-gpu-layers 0 to force CPU.
```

Prompt/runtime behavior:

```text
voice_commands.py now supports action "conversation" with a "response" field.
The command/conversation/unknown decision is handled in one Qwen call, not a two-pass fallback.
The prompt includes examples for land, what-are-you, hello, and unsafe-command rejection.
```

Measured while the vision stack was live:

```text
CPU-only one-shot baseline:
  "what are you": about 14-18 s, peak RSS about 2.0 GB, peak CPU about 250%
  "land the drone now": about 14.3 s, peak RSS about 2.0 GB, peak CPU about 251%

GPU-offloaded one-shot:
  "what are you": 6.44 s, peak RSS about 1.30 GB, peak CPU about 140%
  "land the drone now": 4.52 s, peak RSS about 1.30 GB, peak CPU about 130%
```

Live stream check after GPU offload:

```text
FPS: 29.8 | detections: 2 | persons: 1 | sink: http | hands(mediapipe): 0 | gestures: none | poses: 1 | body: standing
llama_supports_gpu_offload: True
```

System sample after the LLM tests:

```text
RAM: about 3859/7620 MB
Swap: 0/3810 MB
GR3D: 38-62%
VIC: 63-64%
Power VDD_IN: about 14.6-15.0 W
Temps: CPU about 62.7-63.3 C, GPU/TJ about 63.8-63.9 C
```

Notes:

```text
The rebuild changed the local .venv package state and upgraded numpy to 2.2.6.
OpenCV and pyds imports still passed after the rebuild.
pip check still reports an existing protobuf version mismatch for qai-hub.
The CUDA build is an environment change, not a git-tracked code change.
```

### 2026-05-14 Browser Voice UI And Live Resource Snapshot

Goal:

```text
Expose the local Qwen voice command/conversation parser through the DeepStream web page.
Keep the DeepStream process stable while allowing browser text/speech input to ask the local LLM.
```

Implementation:

```text
Added a Start Listening button, typed fallback input, Send button, and result panel to the DeepStream HTTP page.
Added POST /voice-command.
The initial safe endpoint ran voice_commands.py as a subprocess for each request instead of loading llama.cpp/Qwen in-process.
Follow-up added src/drone_ai/apps/voice_llm_server.py as a separate persistent local HTTP service on 127.0.0.1:8091.
src/drone_ai/apps/vision_stream.py calls the warm local server first and falls back to the one-shot subprocess path if the server is unavailable.
The result panel shows heard text, LLM response, parsed action, and confidence.
The endpoint returns JSON only and does not execute flight commands or override the gesture /command state.
```

Important stability note:

```text
Loading the LLM in the DeepStream process during an early POST test destabilized CUDA/DeepStream and triggered a stream restart.
The process boundary keeps llama.cpp/Qwen out of the long-running DeepStream process.
The warm server keeps the model loaded in a separate process, improving response latency while protecting the live video/inference pipeline.
If the warm server crashes or is not started, /voice-command can still use the previous one-shot subprocess fallback.
```

Live endpoint validation:

```bash
curl -fsS --max-time 35 -H 'Content-Type: application/json' -d '{"text":"hello"}' http://127.0.0.1:8080/voice-command
```

```json
{"action":"conversation","value":null,"confidence":0.8,"raw":"hello","response":"Hello. I am listening."}
```

Warm server validation:

```bash
tmux new-session -d -s voice_llm "cd $PROJECT_ROOT && .venv/bin/python -u src/drone_ai/apps/voice_llm_server.py --host 127.0.0.1 --port 8091 2>&1 | tee -a voice_llm_server.log"
curl -fsS http://127.0.0.1:8091/health
curl -fsS -H 'Content-Type: application/json' -d '{"text":"hello"}' http://127.0.0.1:8080/voice-command
```

```text
health: {"ok":true,"model_loaded":true}
warm /voice-command through :8080: elapsed_ms=1653
response: {"action":"conversation","value":null,"confidence":0.8,"raw":"hello","response":"Hello. I am listening."}
```

Day/time handling:

```text
Day/time questions now go through Qwen first.
Qwen returns inspect_time, then src/drone_ai/apps/voice_llm_server.py answers from the Jetson local clock.
This avoids stale model knowledge while keeping Qwen in charge of routing.
Sample through :8080:
  what time is it -> It is 7:32 PM PDT.
  what day is it -> Today is Thursday, May 14, 2026.
```

Resource usage handling:

```text
Resource/performance questions now go through Qwen first.
Qwen returns inspect_resources, then src/drone_ai/apps/voice_llm_server.py reads live local data.
The server reads the stream /stats endpoint, /proc/meminfo, and selected process RSS/CPU values.
Sample through :8080:
  what is the current resource usage ->
  Current usage: stream FPS: 31.0; RAM 4.3 GiB used of 7.4 GiB, 3.1 GiB available, swap 22 MB used; nvargus 14% CPU, 156 MB RSS; DeepStream 84% CPU, 834 MB RSS; MediaPipe hands 68% CPU, 196 MB RSS; voice LLM 16% CPU, 597 MB RSS.
```

Pose/arm measurement handling:

```text
Pose measurement questions now go through Qwen first.
Qwen returns inspect_pose, then src/drone_ai/apps/voice_llm_server.py reads /pose-debug.
The server reads /pose-debug and parses the pose label, facing/orientation labels, and left/right arm out measurements.
Sample through :8080:
  how far out is my right arm ->
  Your right arm out measurement is 416 horizontal and 220 vertical pixels; the current threshold is 200 pixels. Shoulder width is 295 pixels. Current pose is right_arm_out.
```

Live scene summary handling:

```text
"what do you see right now" now goes through Qwen first.
Qwen returns inspect_scene, then src/drone_ai/apps/voice_llm_server.py summarizes /stats and /pose-debug.
The server combines /stats and /pose-debug to report person count, hand count, pose count, body pose, facing label, and FPS.
Sample through :8080:
  Right now, I see 1 person; 0 hand result; 1 pose; body pose right_arm_out; facing front_candidate; stream 30.1 FPS.
```

Qwen-first routing validation:

```text
The routing pass calls Qwen without conversation history for drone commands and live-inspection tool selection.
If Qwen returns conversation, a second history-aware pass or a harmless local memory lookup can answer conversational follow-ups.
Validated through :8080:
  what do you see right now -> inspect_scene -> live scene response
  how far out is my right arm -> inspect_pose -> live arm measurement
  what is the current resource usage -> inspect_resources -> live resource response
  what time is it -> inspect_time -> Jetson clock response
  land the drone now -> land
  my name is operator; what is my name -> Your name is operator.
```

Live stream sample after the web voice UI:

```text
FPS: 30.0 | detections: 4 | persons: 1 | sink: http | auto zoom: hold max zoom-out cropped edge person_h=0.82 none=-1.00 smooth=-1.00 target=0.45-0.65 raw_zoom=2020 requested_zoom=2020 focus=2100 | hands(mediapipe): 0 | gestures: none | poses: 1 | body: right_arm_out
```

Live `/pose-debug` timing sample:

```text
command+overlay+encode: 5.90 ms (60 samples)
jpeg encode: 6.49 ms (30 samples)
command update: 8.53 ms (16 samples)
person lock: 1.75 ms (60 samples)
cpu map+cvtColor: 1.44 ms (60 samples)
overlay draw: 0.71 ms (30 samples)
```

Process resource sample:

```text
src/drone_ai/apps/vision_stream.py: 87.5% CPU, 855 MB RSS
mediapipe_hands_worker.py: 66.5% CPU, 204 MB RSS
nvargus-daemon: 14.2% CPU, 162 MB RSS
src/drone_ai/apps/voice_llm_server.py: 3.6% CPU warm/idle, 607 MB RSS
```

System resource sample:

```text
RAM before warm voice server: 2967/7620 MB
RAM with warm voice server loaded: 3.9/7.4 GiB, 3.3 GiB available
Swap: 22/3810 MB
CPU cores: about 18-39% at 1728 MHz
GR3D: 30-41%
VIC: 20-33%
Power VDD_IN: about 13.4 W
Temps: CPU about 64.7 C, GPU/TJ about 66.1-66.4 C
```

Interpretation:

```text
The live stream remained about 29.4-30 FPS after the voice UI and isolated /voice-command endpoint.
The always-running load is still dominated by DeepStream, MediaPipe hands, and nvargus-daemon.
The warm voice server adds about 607 MB resident memory but keeps normal idle CPU low, and avoids repeated model load latency.
Browser microphone access from Windows may require localhost tunneling or marking the Jetson HTTP origin as secure; typed Send bypasses browser microphone permission.
```

### 2026-05-14 DeepStream Object Label Stability

Change:

```text
Added temporal confirmation for non-person DeepStream object labels in src/drone_ai/apps/vision_stream.py.
Unconfirmed non-person boxes/text are hidden from the browser overlay until the same class is seen consistently.
Person boxes remain immediate so person lock, pose, and command routing are not delayed.
```

Defaults:

```text
--object-label-confirm-frames 10
--object-label-confirm-time 0.0
--object-label-hold 0.5
--object-label-match-iou 0.30
--object-label-min-conf 0.35
```

Validation:

```text
py_compile passed for src/drone_ai/apps/vision_stream.py, voice_commands.py, and src/drone_ai/apps/voice_llm_server.py.
git diff --check passed.
Live /stats after restart:
FPS: 30.0 | detections: 3 | persons: 2 | sink: http | auto zoom: hold max zoom-out cropped edge person_h=0.73 none=-1.00 smooth=-1.00 target=0.45-0.65 raw_zoom=2020 requested_zoom=2020 focus=2100 | hands(mediapipe): 1 | gestures: None:0.85/fingers:0 | poses: 1 | body: left_hand_up
Warm voice server health: ok, model_loaded true.
```

Crash context:

```text
A subsequent interruption was a DeepStream/CUDA pipeline failure, not the non-person label confirmation code.
The log showed nvbufsurftransform_copy.cpp failed in mem copy, then primary_infer reported cudaMemset2DAsync failed with cudaErrorIllegalAddress while converting a buffer.
The GStreamer error was Buffer conversion failed in /GstPipeline:deepstream-object-camera/GstNvInfer:primary_infer.
src/drone_ai/apps/vision_supervisor.py restarted the stream after exit code 255, and the stream returned to about 30 FPS.
Intentional config reloads from code changes show exit code 0 instead.
```

### 2026-05-15 Argus ISP Outdoor Tracking Policy

Change:

```text
DeepStream nvarguscamerasrc now starts with AE/AWB unlocked.
AWB locks after --isp-settle-seconds 2.0.
AE defaults to --isp-ae-lock-mode tracking, so it remains unlocked while searching, locks after a stable tracked target, and unlocks when the target is lost or target ROI brightness shifts.
Edge enhancement was raised to --isp-ee-strength 0.25.
The ISP-tuned camera image feeds the full downstream path: DeepStream inference, person lock/auto zoom, hand/pose processing, and browser MJPEG.
```

Validation:

```text
py_compile passed for src/drone_ai/apps/vision_stream.py.
Restart log printed "Leaving Argus ISP auto controls unlocked for 2.0s before scheduled locks.", "Locked Argus AWB after settle.", and "Locked Argus AE for tracking: stable target."
Detached live stream is running on PID 21384 and serving http://192.0.2.11:8080.
Live /pose-debug showed Argus AE: locked mode=tracking target=visible luma=78.1.
```

Live stream sample:

```text
FPS: 29.8 | detections: 1 | persons: 1 | sink: http | auto zoom: hold max zoom-out cropped edge person_h=0.91 none=-1.00 smooth=-1.00 target=0.45-0.65 raw_zoom=2020 requested_zoom=2020 focus=1870 | tilt: 90 | hands(mediapipe): 0 | gestures: none | poses: 1 | body: right_hand_up
```

Process resource sample:

```text
src/drone_ai/apps/vision_stream.py: 113% CPU, 13.8% MEM
```

System resource sample:

```text
RAM: 3.1 GiB used of 7.4 GiB, 4.1 GiB available, swap 0 B used
tegrastats RAM: 3370-3372/7620 MB, swap 0/3810 MB
CPU cores: about 31-68% at 1113-1728 MHz
GR3D: 58-82%
VIC: 33-74%
Power VDD_IN: about 14.1 W
Temps: CPU about 60.5-60.9 C, GPU/TJ about 61.3-61.5 C
```

### 2026-05-15 DeepStream Web Stream Startup Stability

Crash investigation:

```text
The active camera is detected as /dev/video0, vi-output imx477 10-001a, and V4L2 reports Camera 1: ok.
A short V4L2 raw capture succeeded, confirming the camera can produce frames.
The web stream logs showed the NumPy 2.2.6 warning caused by importing torch through ultralytics.
The normal DeepStream web command uses --pose-backend deepstream, so the legacy Ultralytics pose path should not load at startup.
```

Change:

```text
vision_detectors.py now imports YOLO lazily inside PoseDetector.__init__.
The DeepStream pose + MediaPipe hands startup path no longer loads torch or ultralytics.
HTTP MJPEG client writes now also catch OSError, covering Network is unreachable disconnects.
```

Validation:

```text
py_compile passed for vision_detectors.py and src/drone_ai/apps/vision_stream.py.
Import check with --pose-backend deepstream and --hand-backend mediapipe reported torch_loaded=False and ultralytics_loaded=False.
12-second HTTP startup test on port 8081 started Argus capture, reached about 30 FPS, and exited cleanly under timeout.
No new kernel crash/error appeared after the test.
```
