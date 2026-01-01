import os
import uuid
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FFmpeg path (adjust if needed, or use system ffmpeg on Replit)
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", r"C:\Users\91942\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin")

# Cookies file path for Instagram/TikTok authentication (optional)
COOKIES_FILE = os.environ.get("COOKIES_FILE", "cookies.txt")

# Add ffmpeg to PATH before importing whisper (it needs ffmpeg)
if FFMPEG_PATH:
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import whisper
import yt_dlp

app = FastAPI(title="Video Transcription API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model at startup
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")


class TranscribeRequest(BaseModel):
    url: str


class TranscribeResponse(BaseModel):
    success: bool
    transcript: str | None = None
    error: str | None = None
    duration: float | None = None
    language: str | None = None


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Service is running")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_video(request: TranscribeRequest):
    """
    Transcribe audio from a video URL.

    Supports: YouTube, Instagram Reels, TikTok, Twitter/X videos, and more.
    """
    temp_dir = None

    try:
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, f"{uuid.uuid4()}.%(ext)s")

        # Download audio using yt-dlp Python library
        print(f"Downloading audio from: {request.url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': False,
            'no_warnings': False,
        }

        # Only add ffmpeg_location if path exists
        if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
            ydl_opts['ffmpeg_location'] = FFMPEG_PATH

        # Add cookies file if it exists (required for Instagram, TikTok, etc.)
        if COOKIES_FILE and os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE
            print(f"Using cookies from: {COOKIES_FILE}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([request.url])
        except yt_dlp.utils.DownloadError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download video: {str(e)}"
            )

        # Find the downloaded audio file
        audio_file = None
        for file in os.listdir(temp_dir):
            if file.endswith(('.mp3', '.m4a', '.wav', '.webm', '.opus', '.ogg')):
                audio_file = os.path.join(temp_dir, file)
                break

        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(
                status_code=500,
                detail="Audio file not found after download"
            )

        print(f"Audio downloaded: {audio_file}")

        # Transcribe using Whisper
        print("Starting transcription...")
        result = model.transcribe(audio_file)

        transcript = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # Calculate approximate duration from segments
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0

        print(f"Transcription complete. Language: {language}, Duration: {duration}s")

        return TranscribeResponse(
            success=True,
            transcript=transcript,
            duration=round(duration, 2),
            language=language
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Failed to cleanup temp dir: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Video Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcribe a video from URL",
            "GET /health": "Health check"
        },
        "supported_platforms": [
            "YouTube",
            "Instagram Reels",
            "TikTok",
            "Twitter/X",
            "And many more (via yt-dlp)"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
