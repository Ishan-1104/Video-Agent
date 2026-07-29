import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# A realistic browser User-Agent reduces the chance YouTube flags the
# request as bot traffic.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def download_youtube_audio(url: str) -> str:
    # Use the video ID rather than the title for the output filename.
    # Titles can contain characters (/, :, ?, etc.) that break filesystem
    # paths and make downstream extension-guessing unreliable.
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
        # "android" often returns playable URLs without needing a JS
        # signature solve; falls back to "web" if that client is throttled.
        "extractor_args": {
            "youtube": {"player_client": ["android", "web"]}
        },
        "http_headers": {
            "User-Agent": _USER_AGENT,
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        # FFmpegExtractAudio always outputs .wav regardless of the source
        # container (webm/m4a/opus/etc.), so just swap the extension
        # instead of guessing which original extension to replace.
        filename = os.path.splitext(raw_filename)[0] + ".wav"

    return filename


# data = download_youtube_audio("https://youtu.be/ZVPlLaehjLk?si=HyfEplwQk2BXeW4k")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz
    audio.export(output_path, format="wav")
    return output_path


# data_final = convert_to_wav(data)


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


# print(chunk_audio(data_final))


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
