# Later stream performance and stability log

Sanitized performance-related entries from the project's operating record, covering May–August 2026. The entries describe the system and software stack at the time of each observation; they are not current deployment instructions. Local paths and hostnames have been replaced.

2026-08-23 02:24 PDT - resolved:

- The recurring webstream exits were traced to the JetPack 6.2 / DeepStream
  7.1 `nvvideoconvert` GPU-copy compatibility fault. Each crash began with
  `nvbufsurftransform_copy.cpp:438: Failed in mem copy`, followed by an `nvgpu`
  MMU fault at a null address and CUDA error `700`; the subsequent
  `primary_infer`, pose, and OSD errors were fallout from the poisoned CUDA
  context. RAM, swap, temperatures, Python exceptions, and process ownership
  did not match the failure signature.
- All `nvvideoconvert` elements now use NVIDIA's VIC-copy workaround
  (`copy-hw=2`) through `nvvideoconvert-copy-hw`. Do not restore GPU copies
  (`copy-hw=1`) while this device remains on JetPack 6.2 / DeepStream 7.1.
- The detector and pose TensorRT plans also emitted a cross-device/profile
  warning. They were rebuilt locally, removed from Git tracking, and guarded by
  a launcher preflight. A newly rebuilt plan still encountered the original
  copy fault before the VIC change, so plan mismatch was a secondary deployment
  risk rather than the primary crash cause.
- The post-fix stream started at `02:05:00 PDT`, held `30 FPS` with no new crash
  record or kernel MMU fault through the validation window, and exceeded the
  longest pre-fix run observed that day. The full automated suite passed with
  `127 passed`; both modified shell launchers passed `bash -n`.
- The vision supervisor and DeepStream child are owned by `operator`, not
  `root`. Runtime-user enforcement remains in the launch scripts.

2026-06-25 PDT:

- `drone-stream-control.service` crash was caused by the systemd unit starting the wrong Python target/path for the control page. The service unit should run `$PROJECT_ROOT/.venv/bin/python -m drone_ai.web.control_page`, set `RECORDINGS_DIR=$PROJECT_ROOT/var/recordings`, and serve the control UI on `127.0.0.1:8079`.
- The MediaPipe hand crop overlay delay was caused by `HandGestureDetector` updating `last_crop_overlay` only after the worker returned a result. Because MediaPipe inference is asynchronous, the magenta crop box could follow the previous worker result instead of the current hand/person ROI. `src/drone_ai/vision/detectors.py` now updates the crop overlay when a request is enqueued, so the crop box tracks the current frame/ROI while hand landmarks may still reflect the worker's latest completed inference.
- MAVLink movement commands are now enabled by default in the normal stream path. `scripts/run_web_stream.sh` defaults `MAVLINK_MOTION_CONTROL=1`, `config/stream_profiles/default.json` sets `"mavlink-motion-control": true`, and `src/drone_ai/flight/command_control.py` parser/default args use `mavlink_motion_control=True`.
- The explicit escape hatches remain: `STREAM_PROFILE=no_motion`, `MAVLINK_MOTION_CONTROL=0`, or `--no-mavlink-motion-control` disable actual movement output. MAVLink movement still fails closed behind recent `GUIDED` heartbeat and armed-state checks; follow mode remains separately opt-in with `MAVLINK_FOLLOW=1`.
- Focused validation passed: `bash -n scripts/run_web_stream.sh`, `pytest tests/test_vision_crop.py`, `pytest tests/test_config_loads.py tests/test_imports.py`, and `pytest tests/test_config_loads.py tests/test_command_mapping.py tests/test_gesture_command_map_defaults.py`.

2026-06-24 03:41 PDT:

- `./scripts/run_web_stream.sh status` reports `deepstream_stream` tmux running, HTTP stream listening at `http://jetson.local:8080`, one vision supervisor process, one DeepStream child process, `voice_llm_server` tmux running, and voice LLM listening at `127.0.0.1:8091`.
- Local Piper/TTS is not running by default: no `tts_server` tmux session, no listener on `127.0.0.1:8092`, and zero TTS processes.
- `src/drone_ai/apps/vision_supervisor.py` lock handling should open the lock file without truncating it, acquire `flock()`, then truncate/write the current PID. A competing supervisor must not be able to erase the active supervisor PID before failing to acquire the kernel lock.
- `src/drone_ai/flight/battery.py` normalizes negative MAVLink battery sentinels such as `-1` to `None` before returning JSON or human output. Unknown `battery_remaining`, `current_consumed`, `energy_consumed`, and `time_remaining` must not be spoken or printed as real values.
- `src/drone_ai/recording/transcode.py` must wait on the `gst-launch-1.0` child in the cleanup path even if frame writes fail with `BrokenPipeError` or another write-side `OSError`. Failed transcodes should remove the partial `*.tmp` output before raising.

2026-05-26 05:05 PDT:

- Stream is running under `scripts/run_web_stream.sh` on `jetson.local:8080`; voice LLM is running separately on `127.0.0.1:8091`; persistent Piper TTS is disabled by default and the browser uses Web Speech for spoken responses.
- The recent stream crash at `2026-05-26 04:51:03 PDT` was a DeepStream/TensorRT CUDA path failure, not a Python exception or MediaPipe queue issue: `primary_infer` reported CUDA error `700` / `cudaErrorIllegalAddress`, and the kernel logged an `nvgpu` MMU fault owned by the DeepStream Python process.
- `src/drone_ai/apps/vision_supervisor.py` now starts a `tegrastats` sampler for each stream run and appends the last `120` samples at `1 Hz` to non-zero-exit crash logs. These samples include `GR3D_FREQ`, GPU temperature, RAM, CPU, EMC, VIC, and power rails.
- DeepStream TensorRT engines were rebuilt on this Jetson using the default TensorRT builder optimization level `3`: `deepstream/yolo26s_ds_b1_fp16.engine` rebuilt at `04:43`, and `deepstream/yolo26s_pose_sgie_b1_fp16.engine` rebuilt at `04:36`. Old engines were archived under `deepstream/engine-backups-20260526-044457/`.
- Jetson L4T package mismatch was corrected outside git: `nvidia-l4t-gstreamer` was downgraded from `36.4.7` to `36.4.4-20250616085344` and placed on hold to match the rest of the JetPack/L4T stack.
- Pose is intended to stay enabled. DeepStream pose is filtered to a single selected person ROI before the pose SGIE; skipped person metadata uses class id `80` so the metadata object is preserved while bypassing pose.
- MediaPipe hands now require a current person ROI and run only on the person-derived upper-body crop. The launcher uses side/top crop padding `0.10`, base person pad `0.30`, full-frame probes disabled, and hand result hold reduced to `0.20 s` so stale overlays clear faster.
- MediaPipe hands use one pending frame slot. If the worker is behind, newer JPEG crop requests replace older pending requests; the only unavoidable stale part is an already-running inference plus the configured result hold.
- Recording is capped at `20 FPS` and `1280x720`, with browser-copy transcodes deferred while the stream is live and drained on stop using `CPUQuota=20%`, `nice=19`, and idle `ionice`.
- Qwen voice parsing is kept out of the DeepStream process. The launcher starts `src/drone_ai/apps/voice_llm_server.py` with context `2048`, `4` threads, and GPU layers `-1`. The old subprocess fallback was removed; if the voice LLM server is down, that is treated as a service failure.
- Battery telemetry requests route through MCP and then back through Qwen response generation so spoken battery answers are concise instead of raw device-path telemetry dumps.

2026-05-25 02:16 PDT:

- Stream is intentionally stopped: `deepstream_stream` tmux is not running, port `8080` is not listening, and there are no `src/drone_ai/apps/vision_supervisor.py` or `src/drone_ai/apps/vision_stream.py` processes.
- Latest run log: `logs/vision/vision-stream-20260525-020511-run001.log`.
- Latest run started `2026-05-25 02:05:11 PDT`, stopped cleanly with `return_code=0`, and ran for `73.7 s`.
- Stream configuration in that run: `1920x1080`, `30 FPS`, `sink=http`, browser FPS cap `30`, recording active through `--record-process`, DeepStream pose active, MediaPipe hands active at `20 FPS`, auto zoom active, auto tilt active, blur autofocus disabled.
- TensorRT engines loaded successfully from `deepstream/yolo26s_ds_b1_fp16.engine` and `deepstream/yolo26s_pose_sgie_b1_fp16.engine`.
- FPS samples after startup: `26` samples, average `30.02 FPS`, minimum `28.90 FPS`, maximum `30.90 FPS`.
- Last recording for that run: `recordings/deepstream-20260525-020521.mp4`; final reported recording dropped-frame counter was `166`.
- Browser stream and `/stats` were verified live before stop; `/stream.mjpg` returned multipart JPEG data.
- `src/drone_ai/apps/vision_supervisor.py` now handles stale lock files and `scripts/run_web_stream.sh` uses `logs/vision/vision-stream-$PORT.lock` for future launches instead of `/tmp` to avoid sticky-directory lock permission failures.
