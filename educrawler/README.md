# EduCrawler - Educational Content Crawler

An agentic crawler that extracts educational content from multiple sources (Khan Academy, Byjus, Vedantu, YouTube) and generates study materials for Grade 8 students.

## Project Structure

```
educrawler/
├── src/
│   ├── models/          # Pydantic data models
│   │   ├── request.py   # CrawlRequest, Subject, OutputType
│   │   ├── content.py   # ExtractedContent, VideoTranscript, Concept
│   │   ├── output.py    # StudyMaterial, Questionnaire, Handout
│   │   └── state.py     # CrawlState, ExecutionPlan
│   ├── handlers/        # Source-specific content extractors
│   │   └── youtube.py   # YouTube transcript handler
│   ├── processors/      # Content processors
│   │   └── html_to_md.py # HTML to Markdown converter
│   ├── utils/           # Utilities
│   │   ├── rate_limiter.py # Per-domain rate limiting
│   │   └── retry.py     # Retry decorator with exponential backoff
│   └── renderers/       # Output renderers
│       └── study_material.py # Study material HTML renderer
├── templates/           # Jinja2 templates
├── config/              # Configuration files
│   └── config.yaml      # Main configuration
├── tests/               # Test suite
│   └── test_setup.py    # Setup verification tests
└── requirements.txt     # Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Import Models

```python
from src.models import CrawlRequest, Subject, OutputType

request = CrawlRequest(
    grade=8,
    subject=Subject.PHYSICS,
    topic="Work and Energy",
    output_type=OutputType.STUDY_MATERIAL
)
```

### Extract YouTube Transcripts

```python
import asyncio
from src.handlers.youtube import YouTubeTranscriptHandler

async def main():
    handler = YouTubeTranscriptHandler()
    content = await handler.process("https://www.youtube.com/watch?v=VIDEO_ID")
    
    if content.video_transcripts:
        transcript = content.video_transcripts[0]
        print(f"Title: {transcript.title}")
        print(f"Transcript: {transcript.transcript_text[:200]}...")

asyncio.run(main())
```

### Use Rate Limiter

```python
from src.utils.rate_limiter import RateLimiter

limiter = RateLimiter()
await limiter.acquire("khanacademy.org")  # Waits if needed
```

### Use Retry Decorator

```python
from src.utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=3)
async def fetch_data():
    # Your async function here
    pass
```

## Running Tests

```bash
python -m pytest tests/test_setup.py -v
```

## Features Implemented

- ✅ Complete Pydantic data models for all stages of the pipeline
- ✅ YouTube transcript extraction handler with error handling
- ✅ Per-domain rate limiter with configurable limits
- ✅ Retry decorator with exponential backoff and jitter
- ✅ HTML to Markdown processor
- ✅ Jinja2 template renderer
- ✅ Comprehensive test suite

## Code Quality

- All functions have type hints
- All public functions have docstrings
- Uses async/await for I/O operations
- Explicit error handling (no bare `except:`)
- Logging for important operations

## Next Steps

1. Implement Khan Academy handler
2. Implement Byjus handler
3. Implement Vedantu handler
4. Create content aggregation logic
5. Implement study material generation
6. Add more comprehensive tests
