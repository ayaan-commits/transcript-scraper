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


def generate_summary(transcript: str) -> str:
    """Generate a summary using Groq LLM."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes video transcripts. Provide a concise summary in 2-3 sentences capturing the main points."
                },
                {
                    "role": "user",
                    "content": f"Summarize this transcript:\n\n{transcript[:4000]}"
                }
            ],
            max_tokens=200,
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
            print("Generating AI summary...")
            summary = generate_summary(transcript)

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
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        <!-- Header -->
        <div class="text-center mb-10">
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-3">
                <i class="fas fa-video mr-3 text-purple-400"></i>Video Transcriber
            </h1>
            <p class="text-gray-400 text-lg">
                Transcribe videos from YouTube, Instagram, TikTok & more with AI
            </p>
        </div>

        <!-- Platform badges -->
        <div class="flex justify-center gap-3 mb-8 flex-wrap">
            <span class="px-4 py-2 bg-red-500/20 rounded-full text-red-300 text-sm font-medium">
                <i class="fab fa-youtube mr-2"></i>YouTube
            </span>
            <span class="px-4 py-2 bg-pink-500/20 rounded-full text-pink-300 text-sm font-medium">
                <i class="fab fa-instagram mr-2"></i>Instagram
            </span>
            <span class="px-4 py-2 bg-cyan-500/20 rounded-full text-cyan-300 text-sm font-medium">
                <i class="fab fa-tiktok mr-2"></i>TikTok
            </span>
            <span class="px-4 py-2 bg-blue-500/20 rounded-full text-blue-300 text-sm font-medium">
                <i class="fab fa-twitter mr-2"></i>Twitter/X
            </span>
        </div>

        <!-- Input form -->
        <div class="bg-white/5 backdrop-blur-sm rounded-2xl p-6 mb-6 border border-white/10">
            <form id="transcribeForm">
                <div class="flex flex-col md:flex-row gap-4">
                    <input type="url" id="urlInput" placeholder="Paste video URL here..."
                        class="flex-1 px-5 py-4 rounded-xl bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    <button type="submit" id="submitBtn"
                        class="px-8 py-4 bg-purple-600 hover:bg-purple-500 disabled:bg-purple-800 disabled:cursor-not-allowed rounded-xl text-white font-semibold transition-all flex items-center justify-center gap-2 min-w-[160px]">
                        <i class="fas fa-wand-magic-sparkles"></i>
                        <span>Transcribe</span>
                    </button>
                </div>
            </form>
        </div>

        <!-- Loading state -->
        <div id="loading" class="hidden">
            <div class="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 text-center">
                <div class="relative w-20 h-20 mx-auto mb-4">
                    <div class="absolute inset-0 border-4 border-purple-500/30 rounded-full"></div>
                    <div class="absolute inset-0 border-4 border-purple-500 rounded-full border-t-transparent animate-spin"></div>
                </div>
                <h3 class="text-xl font-semibold text-white mb-2" id="loadingText">Processing your video...</h3>
                <p class="text-gray-400" id="loadingSubtext">Downloading audio and transcribing</p>
            </div>
        </div>

        <!-- Error state -->
        <div id="error" class="hidden">
            <div class="bg-red-500/10 backdrop-blur-sm rounded-2xl p-6 border border-red-500/30">
                <div class="flex items-start gap-4">
                    <div class="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-exclamation-triangle text-red-400"></i>
                    </div>
                    <div>
                        <h4 class="text-red-400 font-semibold mb-1">Error</h4>
                        <p id="errorText" class="text-red-300"></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Result -->
        <div id="result" class="hidden space-y-6">
            <!-- Video info card -->
            <div class="bg-white/5 backdrop-blur-sm rounded-2xl overflow-hidden border border-white/10">
                <div class="flex flex-col md:flex-row">
                    <div class="md:w-72 flex-shrink-0">
                        <img id="thumbnail" src="" alt="Video thumbnail" class="w-full h-full object-cover aspect-video md:aspect-auto">
                    </div>
                    <div class="p-6 flex-1">
                        <h2 id="videoTitle" class="text-xl font-bold text-white mb-3 line-clamp-2"></h2>
                        <div class="flex flex-wrap gap-3 mb-4">
                            <span class="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">
                                <i class="fas fa-check mr-1"></i>Success
                            </span>
                            <span id="langBadge" class="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                                <i class="fas fa-language mr-1"></i><span></span>
                            </span>
                            <span id="durationBadge" class="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm">
                                <i class="fas fa-clock mr-1"></i><span></span>
                            </span>
                        </div>
                        <!-- Download buttons -->
                        <div class="flex flex-wrap gap-2">
                            <button onclick="downloadTXT()" class="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm transition-all flex items-center gap-2">
                                <i class="fas fa-file-alt"></i>TXT
                            </button>
                            <button onclick="downloadSRT()" class="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm transition-all flex items-center gap-2">
                                <i class="fas fa-closed-captioning"></i>SRT
                            </button>
                            <button onclick="downloadVTT()" class="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm transition-all flex items-center gap-2">
                                <i class="fas fa-closed-captioning"></i>VTT
                            </button>
                            <button onclick="copyTranscript()" class="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-white text-sm transition-all flex items-center gap-2">
                                <i class="fas fa-copy"></i><span id="copyText">Copy</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- AI Summary -->
            <div id="summarySection" class="bg-gradient-to-r from-purple-500/10 to-blue-500/10 backdrop-blur-sm rounded-2xl p-6 border border-purple-500/20">
                <h3 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    <i class="fas fa-robot text-purple-400"></i>AI Summary
                </h3>
                <p id="summary" class="text-gray-300 leading-relaxed"></p>
            </div>

            <!-- Transcript -->
            <div class="bg-white/5 backdrop-blur-sm rounded-2xl overflow-hidden border border-white/10">
                <div class="px-6 py-4 border-b border-white/10 flex items-center justify-between">
                    <h3 class="text-lg font-semibold text-white flex items-center gap-2">
                        <i class="fas fa-file-lines text-gray-400"></i>Transcript
                    </h3>
                    <div class="flex gap-2">
                        <button onclick="showPlainTranscript()" id="plainBtn" class="px-3 py-1 bg-purple-600 rounded-lg text-white text-sm">Plain</button>
                        <button onclick="showTimestampTranscript()" id="timestampBtn" class="px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm">Timestamps</button>
                    </div>
                </div>
                <div class="p-6">
                    <div id="transcriptPlain" class="bg-black/30 rounded-xl p-5 max-h-96 overflow-y-auto">
                        <p id="transcript" class="text-gray-200 whitespace-pre-wrap leading-relaxed"></p>
                    </div>
                    <div id="transcriptTimestamps" class="hidden bg-black/30 rounded-xl p-5 max-h-96 overflow-y-auto space-y-3">
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center mt-12 text-gray-500 text-sm">
            <p>Powered by <span class="text-purple-400">Groq Whisper API</span> & <span class="text-purple-400">yt-dlp</span></p>
        </footer>
    </div>

    <script>
        let currentData = null;

        const form = document.getElementById('transcribeForm');
        const urlInput = document.getElementById('urlInput');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorText = document.getElementById('errorText');
        const result = document.getElementById('result');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = urlInput.value.trim();
            if (!url) return;

            loading.classList.remove('hidden');
            error.classList.add('hidden');
            result.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url})
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

        function displayResult(data) {
            // Video info
            document.getElementById('thumbnail').src = data.thumbnail || 'https://via.placeholder.com/480x270?text=No+Thumbnail';
            document.getElementById('videoTitle').textContent = data.title || 'Video';
            document.getElementById('langBadge').querySelector('span').textContent = data.language || 'Unknown';
            document.getElementById('durationBadge').querySelector('span').textContent = data.duration ? data.duration + 's' : '';

            // Summary
            const summarySection = document.getElementById('summarySection');
            if (data.summary) {
                document.getElementById('summary').textContent = data.summary;
                summarySection.classList.remove('hidden');
            } else {
                summarySection.classList.add('hidden');
            }

            // Transcript
            document.getElementById('transcript').textContent = data.transcript;

            // Timestamps
            const timestampContainer = document.getElementById('transcriptTimestamps');
            timestampContainer.innerHTML = '';
            if (data.segments && data.segments.length > 0) {
                data.segments.forEach(seg => {
                    const div = document.createElement('div');
                    div.className = 'flex gap-4 p-3 hover:bg-white/5 rounded-lg transition-all';
                    div.innerHTML = `
                        <span class="text-purple-400 font-mono text-sm whitespace-nowrap">${formatTime(seg.start)}</span>
                        <span class="text-gray-200">${seg.text}</span>
                    `;
                    timestampContainer.appendChild(div);
                });
            }

            result.classList.remove('hidden');
        }

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        function showPlainTranscript() {
            document.getElementById('transcriptPlain').classList.remove('hidden');
            document.getElementById('transcriptTimestamps').classList.add('hidden');
            document.getElementById('plainBtn').className = 'px-3 py-1 bg-purple-600 rounded-lg text-white text-sm';
            document.getElementById('timestampBtn').className = 'px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm';
        }

        function showTimestampTranscript() {
            document.getElementById('transcriptPlain').classList.add('hidden');
            document.getElementById('transcriptTimestamps').classList.remove('hidden');
            document.getElementById('plainBtn').className = 'px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm';
            document.getElementById('timestampBtn').className = 'px-3 py-1 bg-purple-600 rounded-lg text-white text-sm';
        }

        function copyTranscript() {
            navigator.clipboard.writeText(currentData.transcript);
            document.getElementById('copyText').textContent = 'Copied!';
            setTimeout(() => document.getElementById('copyText').textContent = 'Copy', 2000);
        }

        function downloadTXT() {
            const blob = new Blob([currentData.transcript], {type: 'text/plain'});
            downloadBlob(blob, 'transcript.txt');
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
            downloadBlob(blob, 'transcript.srt');
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
            downloadBlob(blob, 'transcript.vtt');
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
