"""
YouTube Downloader Utility

This module provides a function to download video clips from YouTube using
the `yt-dlp` library. It includes a workaround for SSL certificate issues
that can occur on some systems.
"""

import ssl
import certifi
import tempfile
from pathlib import Path
import yt_dlp

# [WORKAROUND] This code block addresses potential SSL certificate validation errors
# when yt-dlp attempts to download content over HTTPS. It forces Python's SSL
# module to use the certificate bundle provided by `certifi`, which is more
# up-to-date than the system's default store on some platforms (e.g., macOS).
ssl._create_default_https_context = (
    lambda *a, **kw: ssl.create_default_context(cafile=certifi.where())
)

def fetch(url: str, max_h: int = 720) -> Path:
    """
    Downloads a YouTube video to a temporary directory.

    This function downloads the best quality MP4 video up to a specified
    maximum height and merges it with the best available audio.

    Args:
        url (str): The URL of the YouTube video to download.
        max_h (int, optional): The maximum desired video height. Defaults to 720.

    Returns:
        Path: The file path to the downloaded MP4 video in a temporary directory.
    """
    # Create a unique temporary directory to store the download.
    tmp_dir = Path(tempfile.mkdtemp())

    # Configure yt-dlp options.
    opts = {
        # Suppress console output from yt-dlp.
        "quiet": True,
        # Specify the desired format: best MP4 video up to max_h, plus best audio.
        # Fallback to the best available if the preferred format is not found.
        "format": (
            f"bestvideo[ext=mp4][height<={max_h}]"
            f"+bestaudio/best[ext=mp4][height<={max_h}]"
        ),
        # Ensure the final output is an MP4 container.
        "merge_output_format": "mp4",
        # Define the output file template.
        "outtmpl": str(tmp_dir / "yt.%(ext)s")
    }

    # Use yt-dlp as a context manager to handle the download.
    with yt_dlp.YoutubeDL(opts) as ydl:
        # Extract video info and trigger the download.
        info = ydl.extract_info(url, download=True)
        # Determine the final filename and return it as a Path object.
        # .with_suffix('.mp4') ensures the correct extension is returned.
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")