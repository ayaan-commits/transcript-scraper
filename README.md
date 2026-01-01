# Video Transcriber

A web application that transcribes videos from URLs (YouTube, Instagram Reels, TikTok, Twitter/X, and more) using OpenAI Whisper.

## Features

- Transcribe videos from multiple platforms:
  - YouTube
  - Instagram Reels
  - TikTok
  - Twitter/X
  - And many more (via yt-dlp)
- Local transcription using OpenAI Whisper (base model)
- Copy transcript to clipboard
- Shows detected language and duration
- Modern UI with Tailwind CSS

## Prerequisites

- Python 3.10+
- Node.js 18+
- ffmpeg (required for audio processing)

### Installing ffmpeg

**Windows:**
```bash
# Using winget
winget install ffmpeg

# Or using chocolatey
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Setup

### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. Paste a video URL from YouTube, Instagram, TikTok, or Twitter/X
3. Click "Transcribe"
4. Wait for the transcription to complete
5. Copy the transcript using the "Copy to Clipboard" button

## API Endpoints

### POST /transcribe

Transcribe a video from a URL.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=..."
}
```

**Response:**
```json
{
  "success": true,
  "transcript": "The transcribed text...",
  "duration": 120.5,
  "language": "en"
}
```

### GET /health

Health check endpoint.

### GET /

API information and supported platforms.

## Tech Stack

- **Backend:** Python, FastAPI, yt-dlp, OpenAI Whisper
- **Frontend:** React, Vite, Tailwind CSS
- **Audio Processing:** ffmpeg

## Notes

- First transcription may be slow as the Whisper model loads
- Longer videos will take more time to process
- Some platforms may require authentication for certain content
- The Whisper base model provides a good balance of speed and accuracy

## Troubleshooting

### "ffmpeg not found" error
Make sure ffmpeg is installed and available in your PATH.

### Video download fails
- Check if the URL is correct and the video is public
- Some videos may be region-locked or require authentication
- Try updating yt-dlp: `pip install --upgrade yt-dlp`

### Slow transcription
- The Whisper model needs to be downloaded on first run
- Consider using a smaller model for faster (but less accurate) results
