import { useState } from 'react'

const API_URL = 'https://7ef5de0ca26790.lhr.life'

function App() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!url.trim()) {
      setError('Please enter a video URL')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url.trim() }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to transcribe video')
      }

      if (data.success) {
        setResult(data)
      } else {
        throw new Error(data.error || 'Transcription failed')
      }
    } catch (err) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = async () => {
    if (result?.transcript) {
      try {
        await navigator.clipboard.writeText(result.transcript)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } catch (err) {
        console.error('Failed to copy:', err)
      }
    }
  }

  const getPlatformIcon = (url) => {
    if (url.includes('youtube.com') || url.includes('youtu.be')) return 'YouTube'
    if (url.includes('instagram.com')) return 'Instagram'
    if (url.includes('tiktok.com')) return 'TikTok'
    if (url.includes('twitter.com') || url.includes('x.com')) return 'Twitter/X'
    return 'Video'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Video Transcriber
          </h1>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Transcribe videos from YouTube, Instagram Reels, TikTok, Twitter/X, and more.
            Powered by OpenAI Whisper.
          </p>
        </div>

        {/* Supported Platforms */}
        <div className="flex justify-center gap-4 mb-8 flex-wrap">
          {['YouTube', 'Instagram', 'TikTok', 'Twitter/X'].map((platform) => (
            <span
              key={platform}
              className="px-4 py-2 bg-white/10 rounded-full text-gray-300 text-sm"
            >
              {platform}
            </span>
          ))}
        </div>

        {/* Main Form */}
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Paste video URL here..."
                className="flex-1 px-6 py-4 rounded-xl bg-white/10 border border-white/20
                         text-white placeholder-gray-400 focus:outline-none focus:ring-2
                         focus:ring-purple-500 focus:border-transparent transition-all"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading}
                className="px-8 py-4 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800
                         disabled:cursor-not-allowed rounded-xl text-white font-semibold
                         transition-all duration-200 flex items-center justify-center gap-2
                         min-w-[160px]"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    <span>Transcribe</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Loading State */}
          {loading && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-8 text-center">
              <div className="inline-block animate-pulse">
                <svg className="w-16 h-16 text-purple-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Processing your video...
              </h3>
              <p className="text-gray-400">
                Downloading audio and transcribing. This may take a minute.
              </p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-6">
              <div className="flex items-start gap-3">
                <svg className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <h4 className="text-red-400 font-semibold mb-1">Error</h4>
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
              {/* Result Header */}
              <div className="bg-white/5 px-6 py-4 border-b border-white/10 flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">
                    Success
                  </span>
                  {result.language && (
                    <span className="text-gray-400 text-sm">
                      Language: <span className="text-white">{result.language.toUpperCase()}</span>
                    </span>
                  )}
                  {result.duration && (
                    <span className="text-gray-400 text-sm">
                      Duration: <span className="text-white">{Math.floor(result.duration / 60)}:{String(Math.floor(result.duration % 60)).padStart(2, '0')}</span>
                    </span>
                  )}
                </div>
                <button
                  onClick={copyToClipboard}
                  className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20
                           rounded-lg text-white transition-all duration-200"
                >
                  {copied ? (
                    <>
                      <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="text-green-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <span>Copy to Clipboard</span>
                    </>
                  )}
                </button>
              </div>

              {/* Transcript */}
              <div className="p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Transcript</h3>
                <div className="bg-black/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <p className="text-gray-200 whitespace-pre-wrap leading-relaxed">
                    {result.transcript}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-500 text-sm">
          <p>Powered by OpenAI Whisper & yt-dlp</p>
        </footer>
      </div>
    </div>
  )
}

export default App
