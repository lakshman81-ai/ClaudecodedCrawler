"""
YouTube transcript extraction handler.

Usage:
    handler = YouTubeTranscriptHandler()
    content = await handler.process("https://youtube.com/watch?v=...")
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

# Import from our models
from src.models import (
    ExtractedContent,
    VideoTranscript,
    ExtractionMetadata,
    SourceType
)

logger = logging.getLogger(__name__)


class YouTubeTranscriptHandler:
    """Handler for extracting YouTube video transcripts."""

    PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)

    def can_handle(self, url: str) -> bool:
        """Check if URL is a YouTube video."""
        patterns = [
            r"youtube\.com/watch\?v=",
            r"youtu\.be/",
            r"youtube\.com/embed/",
        ]
        return any(re.search(p, url) for p in patterns)

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def process(self, url: str) -> ExtractedContent:
        """
        Extract transcript from YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            ExtractedContent with VideoTranscript
        """
        import time
        start_time = time.time()

        video_id = self.extract_video_id(url)
        if not video_id:
            return self._create_error_content(url, "Invalid YouTube URL", start_time)

        # Run synchronous API in thread pool
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._fetch_transcript_sync,
                video_id
            )

            if result["error"]:
                return self._create_error_content(url, result["error"], start_time)

            # Create VideoTranscript model
            transcript = VideoTranscript(
                video_id=video_id,
                video_url=url,
                title=result.get("title", f"YouTube Video {video_id}"),
                transcript_text=result["text"],
                is_auto_generated=result["is_auto_generated"],
                language=result["language"],
                confidence_score=0.95 if not result["is_auto_generated"] else 0.62,
                duration_seconds=int(result.get("duration", 0))
            )

            return ExtractedContent(
                source=SourceType.YOUTUBE,
                url=url,
                topic=f"YouTube: {video_id}",
                video_transcripts=[transcript],
                raw_text=result["text"],
                metadata=ExtractionMetadata(
                    extraction_duration_seconds=time.time() - start_time,
                    content_completeness_score=0.9 if not result["is_auto_generated"] else 0.7,
                    extraction_confidence=transcript.confidence_score,
                    js_rendering_required=False
                )
            )

        except Exception as e:
            logger.error(f"YouTube extraction failed: {e}")
            return self._create_error_content(url, str(e), start_time)

    def _fetch_transcript_sync(self, video_id: str) -> dict:
        """Synchronous transcript fetch (runs in thread pool)."""
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            # Try manual captions first
            transcript = None
            is_auto = False
            language = "en"

            for lang in self.PREFERRED_LANGUAGES:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    if not transcript.is_generated:
                        language = lang
                        break
                except Exception:
                    continue

            # Fall back to auto-generated
            if transcript is None:
                for lang in self.PREFERRED_LANGUAGES:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        is_auto = True
                        language = lang
                        break
                    except Exception:
                        continue

            if transcript is None:
                return {"error": "No transcript available", "text": "", "is_auto_generated": True, "language": "en"}

            # Fetch transcript data
            data = transcript.fetch()
            full_text = " ".join(item["text"] for item in data)
            duration = max((item["start"] + item["duration"] for item in data), default=0)

            return {
                "error": None,
                "text": full_text,
                "is_auto_generated": is_auto or transcript.is_generated,
                "language": language,
                "duration": duration
            }

        except TranscriptsDisabled:
            return {"error": "Transcripts disabled", "text": "", "is_auto_generated": True, "language": "en"}
        except NoTranscriptFound:
            return {"error": "No transcript found", "text": "", "is_auto_generated": True, "language": "en"}
        except VideoUnavailable:
            return {"error": "Video unavailable", "text": "", "is_auto_generated": True, "language": "en"}
        except Exception as e:
            return {"error": str(e), "text": "", "is_auto_generated": True, "language": "en"}

    def _create_error_content(self, url: str, error: str, start_time: float) -> ExtractedContent:
        """Create ExtractedContent for error cases."""
        import time
        return ExtractedContent(
            source=SourceType.YOUTUBE,
            url=url,
            metadata=ExtractionMetadata(
                extraction_duration_seconds=time.time() - start_time,
                content_completeness_score=0.0,
                extraction_confidence=0.0,
                errors=[error]
            )
        )
