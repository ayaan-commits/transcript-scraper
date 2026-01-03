---
title: Video Transcriber
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Video Transcriber

Transcribe videos from YouTube, Instagram Reels, TikTok, Twitter/X, and more using OpenAI Whisper.

## Features

- Supports multiple platforms (YouTube, Instagram, TikTok, Twitter/X)
- Automatic language detection
- Clean web interface
- REST API for integration

## API Usage

```bash
curl -X POST "https://YOUR-SPACE.hf.space/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=..."}'
```

## Tech Stack

- FastAPI
- OpenAI Whisper (base model)
- yt-dlp
- FFmpeg
