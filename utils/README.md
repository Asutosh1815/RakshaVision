# 🛠️ System Utilities & Audio/Alerting HAL

This directory provides utility modules supporting the edge inference pipeline:

- `stream_reader.py`: Non-blocking multi-threaded RTSP and webcam frame grabber preventing video buffer bloat.
- `siren.py`: Dual-engine audible alarm Hardware Abstraction Layer (HAL) supporting Web Audio API synthesizers and local sound hardware.
- `notifier.py`: Asynchronous dispatch client sending instant alert payloads to Telegram bots, webhook endpoints, and SMS gateways.
