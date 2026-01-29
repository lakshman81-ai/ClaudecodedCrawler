"""
EduCrawler YouTube Transcript Handler
=====================================

Handler for extracting YouTube video transcripts using the
youtube-transcript-api library.

This handler is unique because it:
1. Uses an API instead of browser automation
2. Prefers manual captions over auto-generated
3. Requires proxy rotation for cloud environments
4. Can handle multiple languages

Usage:
    from handlers.youtube import YouTubeTranscriptHandler
    
    handler = YouTubeTranscriptHandler()
    content = await handler.process("https://youtube.com/watch?v=...")

Prerequisites:
    pip install youtube-transcript-api
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor

# Note: Actual imports at runtime
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api.formatters import TextFormatter
# from models import ExtractedContent, VideoTranscript, ExtractionMetadata


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class YouTubeConfig:
    """Configuration for YouTube transcript extraction."""
    
    # Language preferences (in order of priority)
    preferred_languages: list[str] = field(
        default_factory=lambda: ["en", "en-US", "en-GB"]
    )
    fallback_to_auto_generated: bool = True
    
    # Proxy configuration (essential for cloud environments)
    use_proxy: bool = False
    proxy_url: Optional[str] = None  # e.g., "http://user:pass@proxy:port"
    
    # Rotating proxy configuration
    proxy_rotation: bool = False
    proxy_list: list[str] = field(default_factory=list)
    
    # Rate limiting
    min_request_interval: float = 1.0
    max_requests_per_minute: int = 30
    
    # Retry settings
    max_retries: int = 2
    backoff_base: float = 2.0
    
    # Content processing
    include_timestamps: bool = True
    segment_duration_threshold: float = 30.0  # Group segments by this duration


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing."""
    
    text: str
    start_seconds: float
    duration_seconds: float
    
    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds
    
    def format_timestamp(self) -> str:
        """Format start time as MM:SS."""
        minutes = int(self.start_seconds // 60)
        seconds = int(self.start_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class TranscriptResult:
    """Result of transcript extraction."""
    
    video_id: str
    video_url: str
    
    # Transcript data
    transcript_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    
    # Metadata
    language: str = "en"
    is_auto_generated: bool = False
    confidence_score: float = 0.95
    
    # Timing
    duration_seconds: float = 0.0
    extraction_duration_seconds: float = 0.0
    
    # Errors
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.transcript_text and not self.error
    
    @property
    def word_count(self) -> int:
        return len(self.transcript_text.split())


# =============================================================================
# YOUTUBE TRANSCRIPT HANDLER
# =============================================================================

class YouTubeTranscriptHandler:
    """
    Handler for extracting YouTube video transcripts.
    
    Uses youtube-transcript-api library which:
    - Works without authentication
    - Bypasses the need for YouTube Data API quota
    - Can retrieve manual and auto-generated captions
    
    IMPORTANT: Cloud provider IPs are frequently blocked by YouTube.
    Use rotating residential proxies for production deployments.
    """
    
    SOURCE_NAME = "youtube"
    SOURCE_TYPE = "youtube"  # Would be SourceType.YOUTUBE
    
    def __init__(self, config: Optional[YouTubeConfig] = None):
        self.config = config or YouTubeConfig()
        self.logger = logging.getLogger("handler.youtube")
        
        # Thread pool for running synchronous API calls
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Rate limiting state
        self._last_request_time = 0.0
        self._request_count = 0
        self._minute_start = time.time()
        
        # Proxy rotation state
        self._proxy_index = 0
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    def can_handle(self, url: str) -> bool:
        """Check if URL is a YouTube video."""
        patterns = [
            r"youtube\.com/watch\?v=",
            r"youtu\.be/",
            r"youtube\.com/embed/",
        ]
        return any(re.search(pattern, url) for pattern in patterns)
    
    async def process(self, url: str) -> "ExtractedContent":
        """
        Main entry point for processing a YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            ExtractedContent with VideoTranscript
        """
        from models import ExtractedContent, VideoTranscript, ExtractionMetadata
        
        if not self.can_handle(url):
            raise ValueError(f"Not a YouTube URL: {url}")
        
        start_time = time.time()
        video_id = self._extract_video_id(url)
        
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")
        
        # Fetch transcript
        result = await self._fetch_transcript(video_id)
        
        if not result.success:
            # Return empty content with error
            return ExtractedContent(
                source=self.SOURCE_TYPE,
                url=url,
                metadata=ExtractionMetadata(
                    extraction_duration_seconds=time.time() - start_time,
                    content_completeness_score=0.0,
                    extraction_confidence=0.0,
                    errors=[result.error or "Unknown error"]
                )
            )
        
        # Convert to VideoTranscript model
        video_transcript = VideoTranscript(
            video_id=video_id,
            video_url=url,
            title="",  # Would need separate API call to get title
            transcript_text=result.transcript_text,
            is_auto_generated=result.is_auto_generated,
            language=result.language,
            confidence_score=result.confidence_score,
            duration_seconds=int(result.duration_seconds),
            key_timestamps=self._extract_key_timestamps(result.segments)
        )
        
        return ExtractedContent(
            source=self.SOURCE_TYPE,
            url=url,
            topic=f"YouTube Video {video_id}",
            video_transcripts=[video_transcript],
            raw_text=result.transcript_text,
            metadata=ExtractionMetadata(
                extraction_duration_seconds=time.time() - start_time,
                content_completeness_score=0.9 if not result.is_auto_generated else 0.7,
                extraction_confidence=result.confidence_score,
                js_rendering_required=False
            )
        )
    
    async def process_batch(self, urls: list[str]) -> list["ExtractedContent"]:
        """
        Process multiple YouTube URLs.
        
        Args:
            urls: List of YouTube URLs
            
        Returns:
            List of ExtractedContent
        """
        tasks = [self.process(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # =========================================================================
    # TRANSCRIPT FETCHING
    # =========================================================================
    
    async def _fetch_transcript(self, video_id: str) -> TranscriptResult:
        """
        Fetch transcript for a video ID.
        
        Uses youtube-transcript-api in a thread pool since it's synchronous.
        """
        # Apply rate limiting
        await self._apply_rate_limit()
        
        start_time = time.time()
        
        # Run synchronous API call in thread pool
        loop = asyncio.get_event_loop()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self._fetch_transcript_sync,
                    video_id
                )
                result.extraction_duration_seconds = time.time() - start_time
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Check for specific errors
                if "blocked" in error_str.lower() or "ip" in error_str.lower():
                    self.logger.warning(f"IP possibly blocked, rotating proxy")
                    self._rotate_proxy()
                
                if attempt < self.config.max_retries:
                    delay = self.config.backoff_base ** attempt
                    self.logger.warning(
                        f"Transcript fetch failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    return TranscriptResult(
                        video_id=video_id,
                        video_url=f"https://youtube.com/watch?v={video_id}",
                        transcript_text="",
                        error=str(e),
                        extraction_duration_seconds=time.time() - start_time
                    )
        
        # Should not reach here
        return TranscriptResult(
            video_id=video_id,
            video_url=f"https://youtube.com/watch?v={video_id}",
            transcript_text="",
            error="Max retries exceeded"
        )
    
    def _fetch_transcript_sync(self, video_id: str) -> TranscriptResult:
        """
        Synchronous transcript fetch using youtube-transcript-api.
        
        This runs in a thread pool.
        """
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable
        )
        
        video_url = f"https://youtube.com/watch?v={video_id}"
        
        # Configure proxy if enabled
        proxies = None
        if self.config.use_proxy and self.config.proxy_url:
            proxies = {
                "http": self.config.proxy_url,
                "https": self.config.proxy_url
            }
        elif self.config.proxy_rotation and self.config.proxy_list:
            proxy = self.config.proxy_list[self._proxy_index]
            proxies = {"http": proxy, "https": proxy}
        
        try:
            # Try to get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                video_id,
                proxies=proxies
            )
            
            # Try preferred languages first (manual captions)
            transcript = None
            is_auto_generated = False
            language = "en"
            
            # First, try manual captions
            for lang in self.config.preferred_languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    if not transcript.is_generated:
                        language = lang
                        break
                except Exception:
                    continue
            
            # Fall back to auto-generated if allowed
            if transcript is None and self.config.fallback_to_auto_generated:
                for lang in self.config.preferred_languages:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        is_auto_generated = True
                        language = lang
                        break
                    except Exception:
                        continue
            
            # Try translation if no direct transcript found
            if transcript is None:
                try:
                    # Get any available transcript and translate
                    available = list(transcript_list)
                    if available:
                        transcript = available[0].translate("en")
                        is_auto_generated = available[0].is_generated
                        language = "en"
                except Exception:
                    pass
            
            if transcript is None:
                return TranscriptResult(
                    video_id=video_id,
                    video_url=video_url,
                    transcript_text="",
                    error="No transcript available in preferred languages"
                )
            
            # Fetch the transcript data
            transcript_data = transcript.fetch()
            
            # Convert to segments
            segments = [
                TranscriptSegment(
                    text=item["text"],
                    start_seconds=item["start"],
                    duration_seconds=item["duration"]
                )
                for item in transcript_data
            ]
            
            # Build full text
            full_text = " ".join(seg.text for seg in segments)
            
            # Calculate total duration
            total_duration = max(
                (seg.end_seconds for seg in segments),
                default=0
            )
            
            # Calculate confidence score
            confidence = 0.95 if not is_auto_generated else 0.62
            
            return TranscriptResult(
                video_id=video_id,
                video_url=video_url,
                transcript_text=full_text,
                segments=segments,
                language=language,
                is_auto_generated=is_auto_generated,
                confidence_score=confidence,
                duration_seconds=total_duration
            )
            
        except TranscriptsDisabled:
            return TranscriptResult(
                video_id=video_id,
                video_url=video_url,
                transcript_text="",
                error="Transcripts are disabled for this video"
            )
        except NoTranscriptFound:
            return TranscriptResult(
                video_id=video_id,
                video_url=video_url,
                transcript_text="",
                error="No transcript found for this video"
            )
        except VideoUnavailable:
            return TranscriptResult(
                video_id=video_id,
                video_url=video_url,
                transcript_text="",
                error="Video is unavailable"
            )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_key_timestamps(
        self,
        segments: list[TranscriptSegment]
    ) -> list[dict]:
        """
        Extract key timestamps from segments.
        
        Groups segments by threshold duration and extracts topic markers.
        """
        if not segments:
            return []
        
        key_timestamps = []
        current_group_start = 0.0
        current_group_text = []
        
        for segment in segments:
            # Check if we should start a new group
            if segment.start_seconds - current_group_start >= self.config.segment_duration_threshold:
                if current_group_text:
                    # Save current group
                    key_timestamps.append({
                        "time_seconds": int(current_group_start),
                        "text": " ".join(current_group_text)[:200],  # Limit length
                        "topic": self._infer_topic(current_group_text)
                    })
                
                # Start new group
                current_group_start = segment.start_seconds
                current_group_text = [segment.text]
            else:
                current_group_text.append(segment.text)
        
        # Don't forget the last group
        if current_group_text:
            key_timestamps.append({
                "time_seconds": int(current_group_start),
                "text": " ".join(current_group_text)[:200],
                "topic": self._infer_topic(current_group_text)
            })
        
        return key_timestamps
    
    def _infer_topic(self, text_parts: list[str]) -> str:
        """Infer a topic from text (simple heuristic)."""
        # This is a placeholder - could use NLP for better topic extraction
        combined = " ".join(text_parts)
        words = combined.split()[:10]
        return " ".join(words) + "..."
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting."""
        # Reset minute counter if needed
        if time.time() - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = time.time()
        
        # Check requests per minute
        if self._request_count >= self.config.max_requests_per_minute:
            wait_time = 60 - (time.time() - self._minute_start)
            if wait_time > 0:
                self.logger.debug(f"Rate limit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._minute_start = time.time()
        
        # Check minimum interval
        time_since_last = time.time() - self._last_request_time
        if time_since_last < self.config.min_request_interval:
            await asyncio.sleep(self.config.min_request_interval - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _rotate_proxy(self) -> None:
        """Rotate to next proxy in list."""
        if self.config.proxy_rotation and self.config.proxy_list:
            self._proxy_index = (self._proxy_index + 1) % len(self.config.proxy_list)
            self.logger.debug(f"Rotated to proxy index {self._proxy_index}")


# =============================================================================
# PROXY CONFIGURATION HELPERS
# =============================================================================

def create_webshare_config(
    username: str,
    password: str,
    proxy_count: int = 10,
    countries: list[str] = None
) -> YouTubeConfig:
    """
    Create configuration for Webshare rotating proxies.
    
    Args:
        username: Webshare username
        password: Webshare password
        proxy_count: Number of proxies to use
        countries: List of country codes (e.g., ["US", "IN"])
        
    Returns:
        YouTubeConfig with proxy settings
    """
    countries = countries or ["US", "IN"]
    
    # Generate proxy URLs
    proxy_list = [
        f"http://{username}:{password}@p.webshare.io:{10000 + i}"
        for i in range(proxy_count)
    ]
    
    return YouTubeConfig(
        use_proxy=True,
        proxy_rotation=True,
        proxy_list=proxy_list
    )


def create_brightdata_config(
    username: str,
    password: str,
    zone: str = "residential"
) -> YouTubeConfig:
    """
    Create configuration for Bright Data (Luminati) proxies.
    
    Args:
        username: Bright Data username
        password: Bright Data password
        zone: Proxy zone (residential, datacenter, etc.)
        
    Returns:
        YouTubeConfig with proxy settings
    """
    proxy_url = f"http://{username}:{password}@brd.superproxy.io:22225"
    
    return YouTubeConfig(
        use_proxy=True,
        proxy_url=proxy_url
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of YouTube transcript handler."""
    
    # Basic usage (no proxy - may be blocked in cloud)
    handler = YouTubeTranscriptHandler()
    
    # Or with Webshare proxy
    # config = create_webshare_config("user", "pass")
    # handler = YouTubeTranscriptHandler(config)
    
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    if handler.can_handle(url):
        content = await handler.process(url)
        
        if content.video_transcripts:
            transcript = content.video_transcripts[0]
            print(f"Video ID: {transcript.video_id}")
            print(f"Language: {transcript.language}")
            print(f"Auto-generated: {transcript.is_auto_generated}")
            print(f"Word count: {transcript.word_count}")
            print(f"\nFirst 500 chars:\n{transcript.transcript_text[:500]}")
        else:
            print(f"Errors: {content.metadata.errors}")


if __name__ == "__main__":
    asyncio.run(example_usage())
