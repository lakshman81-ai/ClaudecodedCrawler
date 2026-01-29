"""Basic tests to verify setup."""

import pytest


def test_models_import():
    """Verify all models can be imported."""
    from src.models import (
        CrawlRequest, Subject, OutputType,
        ExtractedContent, Concept, Formula,
        StudyMaterial, Questionnaire
    )
    assert Subject.PHYSICS.value == "physics"


def test_crawl_request_creation():
    """Test creating a CrawlRequest."""
    from src.models import CrawlRequest, Subject, OutputType

    request = CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work and Energy",
        output_type=OutputType.STUDY_MATERIAL
    )

    assert request.grade == 8
    assert request.subject == Subject.PHYSICS
    assert request.pedagogical_model is not None  # Auto-selected


@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiter basic functionality."""
    from src.utils.rate_limiter import RateLimiter, RateLimitConfig

    config = RateLimitConfig(min_interval_seconds=0.1)
    limiter = RateLimiter(config)

    # First request should be instant
    wait1 = await limiter.acquire("test.com")
    assert wait1 < 0.1

    # Second request should wait
    wait2 = await limiter.acquire("test.com")
    assert wait2 >= 0.05  # Should have waited


@pytest.mark.asyncio
async def test_youtube_handler():
    """Test YouTube handler with real video."""
    from src.handlers.youtube import YouTubeTranscriptHandler

    handler = YouTubeTranscriptHandler()

    # Test URL parsing
    assert handler.can_handle("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert handler.can_handle("https://youtu.be/dQw4w9WgXcQ")
    assert not handler.can_handle("https://google.com")

    # Test video ID extraction
    assert handler.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    # Test actual extraction (uses network)
    content = await handler.process("https://www.youtube.com/watch?v=ZM8ECpBuQYE")

    if content.video_transcripts:
        print(f"✓ Transcript extracted: {len(content.video_transcripts[0].transcript_text)} chars")
    else:
        print(f"! Extraction failed: {content.metadata.errors}")
