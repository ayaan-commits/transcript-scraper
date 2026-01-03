import os
import uuid
import tempfile
import shutil
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from groq import Groq
from typing import Optional

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

app = FastAPI(title="Video Transcription API", version="3.0.0")

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


# ============ Models ============

class TranscribeRequest(BaseModel):
    url: str
    summary_style: str = "brief"  # brief, bullets, takeaways, actions


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    success: bool
    transcript: str | None = None
    segments: list[Segment] | None = None
    summary: str | None = None
    error: str | None = None
    duration: float | None = None
    language: str | None = None
    title: str | None = None
    thumbnail: str | None = None


class MetadataRequest(BaseModel):
    url: str


class MetadataResponse(BaseModel):
    success: bool
    title: str | None = None
    thumbnail: str | None = None
    duration: float | None = None
    channel: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    message: str


# ============ Helper Functions ============

def get_ydl_opts():
    """Get common yt-dlp options with cookies if available."""
    opts = {
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts['cookiefile'] = COOKIES_FILE
    return opts


def get_video_metadata(url: str) -> dict:
    """Extract video metadata without downloading."""
    ydl_opts = get_ydl_opts()
    ydl_opts['skip_download'] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'thumbnail': info.get('thumbnail', ''),
                'duration': info.get('duration', 0),
                'channel': info.get('uploader', info.get('channel', 'Unknown')),
            }
    except Exception as e:
        print(f"Failed to get metadata: {e}")
        return {}


def generate_summary(transcript: str, style: str = "brief") -> str:
    """Generate a summary using Groq LLM with different styles."""

    style_prompts = {
        "brief": "Provide a concise summary in 2-3 sentences capturing the main points.",
        "bullets": "Provide a summary as 4-6 bullet points. Start each point with •",
        "takeaways": "Extract the 3-5 key takeaways from this content. Start each with a number (1., 2., etc.)",
        "actions": "Extract any action items, recommendations, or things the viewer should do. If none exist, summarize the main advice given. Use bullet points starting with •"
    }

    system_prompt = style_prompts.get(style, style_prompts["brief"])

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that summarizes video transcripts. {system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"Summarize this transcript:\n\n{transcript[:4000]}"
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate summary: {e}")
        return ""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT/VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """Convert segments to SRT format."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg['text'].strip())
        srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_vtt(segments: list) -> str:
    """Convert segments to VTT format."""
    vtt_lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = format_vtt_timestamp(seg['start'])
        end = format_vtt_timestamp(seg['end'])
        vtt_lines.append(f"{i}")
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(seg['text'].strip())
        vtt_lines.append("")
    return "\n".join(vtt_lines)


# ============ Endpoints ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Service is running")


@app.post("/metadata", response_model=MetadataResponse)
async def get_metadata(request: MetadataRequest):
    """Get video metadata without transcribing."""
    try:
        metadata = get_video_metadata(request.url)
        if not metadata:
            raise HTTPException(status_code=400, detail="Could not fetch metadata")

        return MetadataResponse(
            success=True,
            title=metadata.get('title'),
            thumbnail=metadata.get('thumbnail'),
            duration=metadata.get('duration'),
            channel=metadata.get('channel'),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_video(request: TranscribeRequest):
    """
    Transcribe audio from a video URL.
    Returns transcript with timestamps, segments, and AI summary.
    """
    temp_dir = None

    try:
        # Get video metadata first
        metadata = get_video_metadata(request.url)

        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, f"{uuid.uuid4()}.%(ext)s")

        # Download audio using yt-dlp Python library
        print(f"Downloading audio from: {request.url}")

        ydl_opts = get_ydl_opts()
        ydl_opts.update({
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        })

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

        # Transcribe using Groq Whisper API with timestamps
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

        # Extract segments with timestamps
        raw_segments = getattr(transcription, 'segments', [])
        segments = [
            Segment(start=seg['start'], end=seg['end'], text=seg['text'])
            for seg in raw_segments
        ] if raw_segments else []

        print(f"Transcription complete. Language: {language}, Duration: {duration}s, Segments: {len(segments)}")

        # Generate AI summary
        summary = ""
        if transcript:
            print(f"Generating AI summary (style: {request.summary_style})...")
            summary = generate_summary(transcript, request.summary_style)

        return TranscribeResponse(
            success=True,
            transcript=transcript,
            segments=segments,
            summary=summary,
            duration=round(duration, 2) if duration else None,
            language=language,
            title=metadata.get('title'),
            thumbnail=metadata.get('thumbnail'),
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


# ============ HTML Template ============

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Transcriber</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        .glass { background: rgba(255,255,255,0.03); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); }
        .glass-strong { background: rgba(255,255,255,0.06); backdrop-filter: blur(30px); border: 1px solid rgba(255,255,255,0.1); }
        .glow { box-shadow: 0 0 40px rgba(147, 51, 234, 0.15); }
        .glow-sm { box-shadow: 0 0 20px rgba(147, 51, 234, 0.1); }
        .highlight { background: linear-gradient(90deg, rgba(250, 204, 21, 0.3), rgba(250, 204, 21, 0.1)); border-radius: 2px; padding: 0 2px; }
        .fade-in { animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .slide-up { animation: slideUp 0.4s ease-out; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .pulse-border { animation: pulseBorder 2s ease-in-out infinite; }
        @keyframes pulseBorder { 0%, 100% { border-color: rgba(147, 51, 234, 0.3); } 50% { border-color: rgba(147, 51, 234, 0.6); } }
        input:focus, select:focus { box-shadow: 0 0 0 2px rgba(147, 51, 234, 0.3); }
        .btn-hover { transition: all 0.2s ease; }
        .btn-hover:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(147, 51, 234, 0.3); }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 3px; }
        ::-webkit-scrollbar-thumb { background: rgba(147, 51, 234, 0.5); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(147, 51, 234, 0.7); }
        .summary-content { white-space: pre-line; }
    </style>
</head>
<body class="min-h-screen bg-[#0a0a0f]">
    <!-- Gradient background -->
    <div class="fixed inset-0 bg-gradient-to-br from-purple-900/20 via-transparent to-blue-900/20 pointer-events-none"></div>
    <div class="fixed top-0 left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl pointer-events-none"></div>
    <div class="fixed bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none"></div>

    <div class="relative container mx-auto px-4 py-8 max-w-5xl">
        <!-- Header -->
        <div class="text-center mb-8 slide-up">
            <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-purple-300 text-sm mb-4">
                <span class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Powered by Groq Whisper
            </div>
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-3 tracking-tight">
                Video <span class="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">Transcriber</span>
            </h1>
            <p class="text-gray-400 text-lg max-w-xl mx-auto">
                Transform any video into searchable, editable text with AI-powered transcription
            </p>
        </div>

        <!-- Platform badges -->
        <div class="flex justify-center gap-2 mb-8 flex-wrap fade-in">
            <span class="px-3 py-1.5 glass rounded-full text-gray-300 text-xs font-medium flex items-center gap-2 hover:bg-white/10 transition-all cursor-default">
                <i class="fab fa-youtube text-red-400"></i>YouTube
            </span>
            <span class="px-3 py-1.5 glass rounded-full text-gray-300 text-xs font-medium flex items-center gap-2 hover:bg-white/10 transition-all cursor-default">
                <i class="fab fa-instagram text-pink-400"></i>Instagram
            </span>
            <span class="px-3 py-1.5 glass rounded-full text-gray-300 text-xs font-medium flex items-center gap-2 hover:bg-white/10 transition-all cursor-default">
                <i class="fab fa-tiktok text-cyan-400"></i>TikTok
            </span>
            <span class="px-3 py-1.5 glass rounded-full text-gray-300 text-xs font-medium flex items-center gap-2 hover:bg-white/10 transition-all cursor-default">
                <i class="fab fa-twitter text-blue-400"></i>Twitter/X
            </span>
        </div>

        <!-- Input form -->
        <div class="glass-strong rounded-2xl p-5 mb-6 glow fade-in">
            <form id="transcribeForm">
                <div class="flex flex-col gap-4">
                    <!-- URL Input Row -->
                    <div class="flex flex-col md:flex-row gap-3">
                        <div class="flex-1 relative">
                            <i class="fas fa-link absolute left-4 top-1/2 -translate-y-1/2 text-gray-500"></i>
                            <input type="url" id="urlInput" placeholder="Paste video URL here..."
                                class="w-full pl-11 pr-4 py-3.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-all">
                        </div>
                        <button type="submit" id="submitBtn"
                            class="px-6 py-3.5 bg-gradient-to-r from-purple-600 to-purple-500 hover:from-purple-500 hover:to-purple-400 disabled:from-purple-800 disabled:to-purple-700 disabled:cursor-not-allowed rounded-xl text-white font-semibold transition-all flex items-center justify-center gap-2 min-w-[150px] btn-hover">
                            <i class="fas fa-wand-magic-sparkles"></i>
                            <span>Transcribe</span>
                        </button>
                    </div>
                    <!-- Options Row -->
                    <div class="flex flex-col sm:flex-row gap-3">
                        <div class="flex-1">
                            <label class="block text-xs text-gray-500 mb-1.5 ml-1">Summary Style</label>
                            <select id="summaryStyle" class="w-full px-4 py-2.5 rounded-lg bg-white/5 border border-white/10 text-gray-300 focus:outline-none focus:border-purple-500/50 transition-all text-sm cursor-pointer">
                                <option value="brief">Brief Summary</option>
                                <option value="bullets">Bullet Points</option>
                                <option value="takeaways">Key Takeaways</option>
                                <option value="actions">Action Items</option>
                            </select>
                        </div>
                    </div>
                </div>
            </form>
        </div>

        <!-- Loading state -->
        <div id="loading" class="hidden fade-in">
            <div class="glass-strong rounded-2xl p-10 text-center glow">
                <div class="relative w-16 h-16 mx-auto mb-5">
                    <div class="absolute inset-0 border-3 border-purple-500/20 rounded-full"></div>
                    <div class="absolute inset-0 border-3 border-purple-500 rounded-full border-t-transparent animate-spin"></div>
                    <i class="fas fa-microphone-alt absolute inset-0 flex items-center justify-center text-purple-400 text-lg"></i>
                </div>
                <h3 class="text-lg font-semibold text-white mb-2" id="loadingText">Processing your video...</h3>
                <p class="text-gray-500 text-sm" id="loadingSubtext">Downloading audio and transcribing with AI</p>
                <div class="mt-4 flex justify-center gap-1">
                    <div class="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style="animation-delay: 0s"></div>
                    <div class="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <div class="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                </div>
            </div>
        </div>

        <!-- Error state -->
        <div id="error" class="hidden fade-in">
            <div class="glass rounded-2xl p-5 border-red-500/30 bg-red-500/5">
                <div class="flex items-start gap-4">
                    <div class="w-10 h-10 bg-red-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-exclamation-triangle text-red-400"></i>
                    </div>
                    <div>
                        <h4 class="text-red-400 font-semibold mb-1">Something went wrong</h4>
                        <p id="errorText" class="text-red-300/80 text-sm"></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Result -->
        <div id="result" class="hidden slide-up">
            <div class="glass-strong rounded-2xl overflow-hidden glow">
                <!-- Video Header -->
                <div class="flex flex-col md:flex-row">
                    <!-- Thumbnail -->
                    <div class="md:w-72 flex-shrink-0 relative group">
                        <img id="thumbnail" src="" alt="Video thumbnail" class="w-full h-48 md:h-full object-cover">
                        <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
                        <div class="absolute bottom-3 left-3 right-3">
                            <span id="durationBadge" class="px-2 py-1 bg-black/60 backdrop-blur-sm text-white rounded text-xs">
                                <i class="fas fa-clock mr-1"></i><span></span>
                            </span>
                        </div>
                    </div>

                    <!-- Info -->
                    <div class="flex-1 p-5">
                        <div class="flex items-start justify-between gap-4 mb-4">
                            <div>
                                <h2 id="videoTitle" class="text-xl font-bold text-white mb-2 line-clamp-2"></h2>
                                <div class="flex flex-wrap gap-2">
                                    <span class="px-2.5 py-1 bg-green-500/20 text-green-400 rounded-lg text-xs font-medium">
                                        <i class="fas fa-check-circle mr-1"></i>Transcribed
                                    </span>
                                    <span id="langBadge" class="px-2.5 py-1 bg-blue-500/20 text-blue-400 rounded-lg text-xs">
                                        <i class="fas fa-globe mr-1"></i><span></span>
                                    </span>
                                </div>
                            </div>
                        </div>

                        <!-- Export buttons -->
                        <div class="flex flex-wrap gap-2">
                            <span class="text-xs text-gray-500 w-full mb-1">Export as:</span>
                            <button onclick="downloadTXT()" class="px-3 py-2 glass hover:bg-white/10 rounded-lg text-gray-300 text-xs transition-all flex items-center gap-2 btn-hover">
                                <i class="fas fa-file-alt text-gray-400"></i>TXT
                            </button>
                            <button onclick="downloadSRT()" class="px-3 py-2 glass hover:bg-white/10 rounded-lg text-gray-300 text-xs transition-all flex items-center gap-2 btn-hover">
                                <i class="fas fa-closed-captioning text-yellow-400"></i>SRT
                            </button>
                            <button onclick="downloadVTT()" class="px-3 py-2 glass hover:bg-white/10 rounded-lg text-gray-300 text-xs transition-all flex items-center gap-2 btn-hover">
                                <i class="fas fa-closed-captioning text-blue-400"></i>VTT
                            </button>
                            <button onclick="downloadJSON()" class="px-3 py-2 glass hover:bg-white/10 rounded-lg text-gray-300 text-xs transition-all flex items-center gap-2 btn-hover">
                                <i class="fas fa-code text-green-400"></i>JSON
                            </button>
                            <button onclick="copyTranscript()" class="px-3 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-white text-xs transition-all flex items-center gap-2 btn-hover">
                                <i class="fas fa-copy"></i><span id="copyText">Copy</span>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- AI Summary -->
                <div id="summarySection" class="px-5 py-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border-t border-b border-white/5">
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-robot text-purple-400 text-sm"></i>
                        </div>
                        <div class="flex-1">
                            <h4 class="text-xs font-semibold text-purple-300 uppercase tracking-wider mb-1">AI Summary</h4>
                            <p id="summary" class="text-gray-300 text-sm leading-relaxed summary-content"></p>
                        </div>
                    </div>
                </div>

                <!-- Transcript Section -->
                <div class="p-5">
                    <!-- Search & View Toggle -->
                    <div class="flex flex-col sm:flex-row gap-3 mb-4">
                        <div class="flex-1 relative">
                            <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm"></i>
                            <input type="text" id="searchInput" placeholder="Search transcript..."
                                class="w-full pl-9 pr-4 py-2.5 rounded-lg bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-all text-sm">
                            <span id="searchCount" class="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500 hidden"></span>
                        </div>
                        <div class="flex gap-1 bg-white/5 p-1 rounded-lg">
                            <button onclick="showPlainTranscript()" id="plainBtn" class="px-4 py-2 bg-purple-600 rounded-md text-white text-xs font-medium transition-all">Plain</button>
                            <button onclick="showTimestampTranscript()" id="timestampBtn" class="px-4 py-2 hover:bg-white/10 rounded-md text-gray-400 text-xs font-medium transition-all">Timestamps</button>
                        </div>
                    </div>

                    <!-- Transcript Content -->
                    <div class="bg-black/30 rounded-xl border border-white/5">
                        <div id="transcriptPlain" class="h-72 overflow-y-auto p-4">
                            <p id="transcript" class="text-gray-300 text-sm whitespace-pre-wrap leading-relaxed"></p>
                        </div>
                        <div id="transcriptTimestamps" class="hidden h-72 overflow-y-auto p-4 space-y-1">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center mt-10 text-gray-600 text-xs">
            <p>Built with <i class="fas fa-heart text-purple-500"></i> using <span class="text-gray-500">Groq Whisper</span> & <span class="text-gray-500">yt-dlp</span></p>
        </footer>
    </div>

    <script>
        let currentData = null;

        const form = document.getElementById('transcribeForm');
        const urlInput = document.getElementById('urlInput');
        const submitBtn = document.getElementById('submitBtn');
        const summaryStyle = document.getElementById('summaryStyle');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorText = document.getElementById('errorText');
        const result = document.getElementById('result');
        const searchInput = document.getElementById('searchInput');
        const searchCount = document.getElementById('searchCount');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = urlInput.value.trim();
            if (!url) return;

            loading.classList.remove('hidden');
            error.classList.add('hidden');
            result.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i><span>Processing...</span>';

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        url: url,
                        summary_style: summaryStyle.value
                    })
                });
                const data = await response.json();

                if (data.success) {
                    currentData = data;
                    displayResult(data);
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
                submitBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i><span>Transcribe</span>';
            }
        });

        // Search functionality
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim().toLowerCase();
            highlightSearch(query);
        });

        function highlightSearch(query) {
            const transcriptEl = document.getElementById('transcript');
            const timestampContainer = document.getElementById('transcriptTimestamps');

            if (!currentData) return;

            if (!query) {
                transcriptEl.innerHTML = escapeHtml(currentData.transcript);
                searchCount.classList.add('hidden');
                // Reset timestamp view
                if (currentData.segments) {
                    renderTimestamps(currentData.segments);
                }
                return;
            }

            // Highlight in plain transcript
            const text = currentData.transcript;
            const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
            const matches = text.match(regex);
            const count = matches ? matches.length : 0;

            searchCount.textContent = `${count} found`;
            searchCount.classList.remove('hidden');

            transcriptEl.innerHTML = escapeHtml(text).replace(
                new RegExp(`(${escapeRegex(escapeHtml(query))})`, 'gi'),
                '<mark class="highlight">$1</mark>'
            );

            // Highlight in timestamps
            if (currentData.segments) {
                renderTimestamps(currentData.segments, query);
            }
        }

        function renderTimestamps(segments, query = '') {
            const container = document.getElementById('transcriptTimestamps');
            container.innerHTML = '';

            segments.forEach(seg => {
                const div = document.createElement('div');
                div.className = 'flex gap-3 p-2.5 hover:bg-white/5 rounded-lg transition-all cursor-default';

                let textHtml = escapeHtml(seg.text);
                if (query) {
                    textHtml = textHtml.replace(
                        new RegExp(`(${escapeRegex(escapeHtml(query))})`, 'gi'),
                        '<mark class="highlight">$1</mark>'
                    );
                }

                div.innerHTML = `
                    <span class="text-purple-400 font-mono text-xs bg-purple-500/10 px-2 py-1 rounded whitespace-nowrap h-fit">${formatTime(seg.start)}</span>
                    <span class="text-gray-300 text-sm">${textHtml}</span>
                `;
                container.appendChild(div);
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function escapeRegex(string) {
            return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }

        function displayResult(data) {
            document.getElementById('thumbnail').src = data.thumbnail || 'https://via.placeholder.com/480x270/1a1a2e/6366f1?text=No+Thumbnail';
            document.getElementById('videoTitle').textContent = data.title || 'Video';
            document.getElementById('langBadge').querySelector('span').textContent = data.language ? data.language.toUpperCase() : 'N/A';
            document.getElementById('durationBadge').querySelector('span').textContent = data.duration ? formatDuration(data.duration) : '';

            const summarySection = document.getElementById('summarySection');
            if (data.summary) {
                document.getElementById('summary').textContent = data.summary;
                summarySection.classList.remove('hidden');
            } else {
                summarySection.classList.add('hidden');
            }

            document.getElementById('transcript').textContent = data.transcript;
            searchInput.value = '';
            searchCount.classList.add('hidden');

            if (data.segments && data.segments.length > 0) {
                renderTimestamps(data.segments);
            }

            result.classList.remove('hidden');
        }

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        function formatDuration(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            if (mins >= 60) {
                const hrs = Math.floor(mins / 60);
                const remainMins = mins % 60;
                return `${hrs}:${remainMins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        function showPlainTranscript() {
            document.getElementById('transcriptPlain').classList.remove('hidden');
            document.getElementById('transcriptTimestamps').classList.add('hidden');
            document.getElementById('plainBtn').className = 'px-4 py-2 bg-purple-600 rounded-md text-white text-xs font-medium transition-all';
            document.getElementById('timestampBtn').className = 'px-4 py-2 hover:bg-white/10 rounded-md text-gray-400 text-xs font-medium transition-all';
        }

        function showTimestampTranscript() {
            document.getElementById('transcriptPlain').classList.add('hidden');
            document.getElementById('transcriptTimestamps').classList.remove('hidden');
            document.getElementById('plainBtn').className = 'px-4 py-2 hover:bg-white/10 rounded-md text-gray-400 text-xs font-medium transition-all';
            document.getElementById('timestampBtn').className = 'px-4 py-2 bg-purple-600 rounded-md text-white text-xs font-medium transition-all';
        }

        function copyTranscript() {
            navigator.clipboard.writeText(currentData.transcript);
            const btn = document.getElementById('copyText');
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 2000);
        }

        function downloadTXT() {
            const blob = new Blob([currentData.transcript], {type: 'text/plain'});
            downloadBlob(blob, `${sanitizeFilename(currentData.title)}_transcript.txt`);
        }

        function downloadJSON() {
            const exportData = {
                title: currentData.title,
                duration: currentData.duration,
                language: currentData.language,
                transcript: currentData.transcript,
                summary: currentData.summary,
                segments: currentData.segments,
                exported_at: new Date().toISOString()
            };
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
            downloadBlob(blob, `${sanitizeFilename(currentData.title)}_transcript.json`);
        }

        function downloadSRT() {
            if (!currentData.segments || currentData.segments.length === 0) {
                alert('No timestamp data available');
                return;
            }
            let srt = '';
            currentData.segments.forEach((seg, i) => {
                srt += `${i + 1}\\n`;
                srt += `${formatSRTTime(seg.start)} --> ${formatSRTTime(seg.end)}\\n`;
                srt += `${seg.text.trim()}\\n\\n`;
            });
            const blob = new Blob([srt], {type: 'text/plain'});
            downloadBlob(blob, `${sanitizeFilename(currentData.title)}_transcript.srt`);
        }

        function downloadVTT() {
            if (!currentData.segments || currentData.segments.length === 0) {
                alert('No timestamp data available');
                return;
            }
            let vtt = 'WEBVTT\\n\\n';
            currentData.segments.forEach((seg, i) => {
                vtt += `${i + 1}\\n`;
                vtt += `${formatVTTTime(seg.start)} --> ${formatVTTTime(seg.end)}\\n`;
                vtt += `${seg.text.trim()}\\n\\n`;
            });
            const blob = new Blob([vtt], {type: 'text/vtt'});
            downloadBlob(blob, `${sanitizeFilename(currentData.title)}_transcript.vtt`);
        }

        function sanitizeFilename(name) {
            return (name || 'video').replace(/[^a-z0-9]/gi, '_').substring(0, 50);
        }

        function formatSRTTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')},${ms.toString().padStart(3,'0')}`;
        }

        function formatVTTTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}.${ms.toString().padStart(3,'0')}`;
        }

        function downloadBlob(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
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
