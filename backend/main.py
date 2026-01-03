import os
import uuid
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

# Cookies file path for Instagram/TikTok authentication (optional)
# Render mounts secret files at /etc/secrets/ (read-only)
COOKIES_SOURCE = os.environ.get("COOKIES_FILE", "/etc/secrets/cookies.txt")
# Fallback to local cookies.txt for local development
if not os.path.exists(COOKIES_SOURCE):
    COOKIES_SOURCE = "cookies.txt"

# Copy cookies to writable location (yt-dlp may try to update them)
COOKIES_FILE = "/tmp/cookies.txt"
if os.path.exists(COOKIES_SOURCE):
    shutil.copy(COOKIES_SOURCE, COOKIES_FILE)
    print(f"Copied cookies from {COOKIES_SOURCE} to {COOKIES_FILE}")

# Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

import yt_dlp

app = FastAPI(title="Video Transcription API", version="2.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
print("Groq Whisper API initialized!")

# Log cookies file status
if os.path.exists(COOKIES_FILE):
    print(f"Cookies file ready at: {COOKIES_FILE}")
else:
    print(f"No cookies file found (checked: {COOKIES_SOURCE})")


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
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

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

        # Transcribe using Groq Whisper API
        print("Starting transcription with Groq Whisper...")

        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file), f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )

        transcript = transcription.text.strip()
        language = getattr(transcription, 'language', 'unknown')
        duration = getattr(transcription, 'duration', 0)

        print(f"Transcription complete. Language: {language}, Duration: {duration}s")

        return TranscribeResponse(
            success=True,
            transcript=transcript,
            duration=round(duration, 2) if duration else None,
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


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Transcriber</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
    <div class="container mx-auto px-4 py-12">
        <div class="text-center mb-12">
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-4">Video Transcriber</h1>
            <p class="text-gray-300 text-lg max-w-2xl mx-auto">
                Transcribe videos from YouTube, Instagram Reels, TikTok, Twitter/X, and more.
            </p>
        </div>

        <div class="flex justify-center gap-4 mb-8 flex-wrap">
            <span class="px-4 py-2 bg-white/10 rounded-full text-gray-300 text-sm">YouTube</span>
            <span class="px-4 py-2 bg-white/10 rounded-full text-gray-300 text-sm">Instagram</span>
            <span class="px-4 py-2 bg-white/10 rounded-full text-gray-300 text-sm">TikTok</span>
            <span class="px-4 py-2 bg-white/10 rounded-full text-gray-300 text-sm">Twitter/X</span>
        </div>

        <div class="max-w-3xl mx-auto">
            <form id="transcribeForm" class="mb-8">
                <div class="flex flex-col md:flex-row gap-4">
                    <input type="url" id="urlInput" placeholder="Paste video URL here..."
                        class="flex-1 px-6 py-4 rounded-xl bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500">
                    <button type="submit" id="submitBtn"
                        class="px-8 py-4 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 rounded-xl text-white font-semibold transition-all min-w-[160px]">
                        Transcribe
                    </button>
                </div>
            </form>

            <div id="loading" class="hidden bg-white/5 border border-white/10 rounded-xl p-8 text-center mb-6">
                <div class="animate-pulse">
                    <svg class="w-16 h-16 text-purple-400 mx-auto mb-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
                <h3 class="text-xl font-semibold text-white mb-2">Processing your video...</h3>
                <p class="text-gray-400">Downloading audio and transcribing. This may take a minute.</p>
            </div>

            <div id="error" class="hidden bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-6">
                <h4 class="text-red-400 font-semibold mb-1">Error</h4>
                <p id="errorText" class="text-red-300"></p>
            </div>

            <div id="result" class="hidden bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                <div class="bg-white/5 px-6 py-4 border-b border-white/10 flex items-center justify-between flex-wrap gap-4">
                    <div class="flex items-center gap-4">
                        <span class="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">Success</span>
                        <span id="langBadge" class="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm"></span>
                        <span id="durationBadge" class="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm"></span>
                    </div>
                    <button onclick="copyTranscript()" class="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white transition-all">
                        <span id="copyText">Copy to Clipboard</span>
                    </button>
                </div>
                <div class="p-6">
                    <h3 class="text-lg font-semibold text-white mb-4">Transcript</h3>
                    <div class="bg-black/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                        <p id="transcript" class="text-gray-200 whitespace-pre-wrap leading-relaxed"></p>
                    </div>
                </div>
            </div>
        </div>

        <footer class="text-center mt-16 text-gray-500 text-sm">
            <p>Powered by Groq Whisper API & yt-dlp</p>
        </footer>
    </div>

    <script>
        const form = document.getElementById('transcribeForm');
        const urlInput = document.getElementById('urlInput');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorText = document.getElementById('errorText');
        const result = document.getElementById('result');
        const transcript = document.getElementById('transcript');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = urlInput.value.trim();
            if (!url) return;

            loading.classList.remove('hidden');
            error.classList.add('hidden');
            result.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url})
                });
                const data = await response.json();

                if (data.success) {
                    transcript.textContent = data.transcript;
                    document.getElementById('langBadge').textContent = data.language || '';
                    document.getElementById('durationBadge').textContent = data.duration ? data.duration + 's' : '';
                    result.classList.remove('hidden');
                } else {
                    errorText.textContent = data.detail || data.error || 'Transcription failed';
                    error.classList.remove('hidden');
                }
            } catch (err) {
                errorText.textContent = err.message || 'An error occurred';
                error.classList.remove('hidden');
            } finally {
                loading.classList.add('hidden');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Transcribe';
            }
        });

        function copyTranscript() {
            navigator.clipboard.writeText(transcript.textContent);
            document.getElementById('copyText').textContent = 'Copied!';
            setTimeout(() => document.getElementById('copyText').textContent = 'Copy to Clipboard', 2000);
        }
    </script>
</body>
</html>
'''


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    return HTML_TEMPLATE


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
