"""
EduCrawler Request Models
=========================

Pydantic models for user input validation and crawl configuration.

Usage:
    from models.request import CrawlRequest, CrawlConfig
    
    request = CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work Done",
        subtopics=["Kinetic Energy", "Potential Energy"],
        output_type=OutputType.STUDY_MATERIAL
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class Subject(str, Enum):
    """Supported subjects for Grade 8 curriculum."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    HISTORY = "history"
    SOCIAL_STUDIES = "social_studies"
    ENGLISH = "english"


class OutputType(str, Enum):
    """Types of educational output documents."""
    STUDY_MATERIAL = "study_material"
    QUESTIONNAIRE = "questionnaire"
    HANDOUT = "handout"


class SourceType(str, Enum):
    """Available content sources."""
    KHAN_ACADEMY = "khan_academy"
    BYJUS = "byjus"
    VEDANTU = "vedantu"
    YOUTUBE = "youtube"
    GOOGLE_SEARCH = "google_search"


class QueryType(str, Enum):
    """Classification of query intent for source prioritization."""
    CONCEPTUAL = "conceptual"      # "What is...", "Explain..."
    PROBLEM_BASED = "problem"      # "Calculate...", "Solve..."
    VIDEO_REQUEST = "video"        # "Show me...", "Video of..."
    COMPREHENSIVE = "comprehensive" # Multiple aspects needed


class PedagogicalModel(str, Enum):
    """Pedagogical frameworks for content organization."""
    FIVE_E = "5e"           # Engage, Explore, Explain, Elaborate, Evaluate
    LES = "les"             # Launch, Explore, Summarize
    NARRATIVE = "narrative" # Chronological/thematic for History
    RULE_EXAMPLE = "rule_example"  # Rule → Example → Application for English


class SourcePriority(BaseModel):
    """Configuration for source prioritization based on query type."""
    
    source: SourceType
    priority: int = Field(ge=1, le=10, description="1 = highest priority")
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=5)
    required: bool = Field(default=False, description="Fail if this source fails")
    
    class Config:
        frozen = True


class CrawlConfig(BaseModel):
    """Execution configuration for the crawler."""
    
    # Parallelism
    max_parallel_sources: int = Field(default=3, ge=1, le=5)
    max_parallel_pages_per_source: int = Field(default=2, ge=1, le=5)
    
    # Timeouts
    global_timeout_seconds: int = Field(default=300, ge=60, le=600)
    page_load_timeout_seconds: int = Field(default=30, ge=10, le=60)
    element_wait_timeout_seconds: int = Field(default=10, ge=5, le=30)
    
    # Rate Limiting
    min_request_interval_seconds: float = Field(default=2.0, ge=1.0, le=10.0)
    max_requests_per_minute_per_domain: int = Field(default=20, ge=5, le=60)
    
    # Retry Configuration
    retry_backoff_base: float = Field(default=2.0)
    retry_backoff_max_seconds: float = Field(default=32.0)
    retry_jitter_range: tuple[float, float] = Field(default=(0.0, 0.5))
    
    # Quality Gates
    min_content_quality_score: float = Field(default=0.6, ge=0.0, le=1.0)
    deduplication_similarity_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    
    # NotebookLM
    notebooklm_enabled: bool = Field(default=True)
    notebooklm_auth_state_path: str = Field(default="notebooklm_auth.json")
    notebooklm_max_sources: int = Field(default=10, ge=1, le=50)
    
    # Output
    include_source_attribution: bool = Field(default=True)
    generate_answer_key: bool = Field(default=True)  # For questionnaires


class YouTubeLink(BaseModel):
    """Validated YouTube video link."""
    
    url: str
    video_id: str = Field(default="", description="Extracted video ID")
    title: Optional[str] = Field(default=None)
    
    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        """Ensure URL is a valid YouTube link."""
        valid_patterns = [
            "youtube.com/watch?v=",
            "youtu.be/",
            "youtube.com/embed/",
        ]
        if not any(pattern in v for pattern in valid_patterns):
            raise ValueError(f"Invalid YouTube URL: {v}")
        return v
    
    @model_validator(mode="after")
    def extract_video_id(self) -> "YouTubeLink":
        """Extract video ID from URL."""
        import re
        patterns = [
            r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.url)
            if match:
                object.__setattr__(self, "video_id", match.group(1))
                break
        return self


class CrawlRequest(BaseModel):
    """
    Primary input model for educational content crawling.
    
    Example:
        request = CrawlRequest(
            grade=8,
            subject=Subject.PHYSICS,
            topic="Work, Energy and Power",
            subtopics=["Work Done", "Kinetic Energy", "Potential Energy"],
            youtube_links=[
                YouTubeLink(url="https://youtube.com/watch?v=EXAMPLE1")
            ],
            output_type=OutputType.STUDY_MATERIAL
        )
    """
    
    # Request Identification
    request_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Educational Context
    grade: int = Field(ge=1, le=12, description="Student grade level")
    subject: Subject
    topic: str = Field(min_length=2, max_length=200)
    subtopics: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific subtopics to cover"
    )
    
    # Content Sources
    youtube_links: list[YouTubeLink] = Field(
        default_factory=list,
        max_length=5,
        description="Specific YouTube videos to include"
    )
    custom_urls: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Additional URLs to scrape"
    )
    
    # Source Control
    enabled_sources: list[SourceType] = Field(
        default_factory=lambda: [
            SourceType.KHAN_ACADEMY,
            SourceType.BYJUS,
            SourceType.VEDANTU,
            SourceType.YOUTUBE,
        ]
    )
    source_priorities: list[SourcePriority] = Field(
        default_factory=list,
        description="Custom source priorities (auto-generated if empty)"
    )
    include_google_fallback: bool = Field(
        default=True,
        description="Use Google Search as fallback"
    )
    
    # Output Configuration
    output_type: OutputType = Field(default=OutputType.STUDY_MATERIAL)
    pedagogical_model: Optional[PedagogicalModel] = Field(
        default=None,
        description="Auto-selected based on subject if None"
    )
    
    # Execution Configuration
    config: CrawlConfig = Field(default_factory=CrawlConfig)
    
    @field_validator("topic")
    @classmethod
    def clean_topic(cls, v: str) -> str:
        """Normalize topic string."""
        return " ".join(v.strip().split())
    
    @field_validator("subtopics")
    @classmethod
    def clean_subtopics(cls, v: list[str]) -> list[str]:
        """Normalize subtopic strings."""
        return [" ".join(s.strip().split()) for s in v if s.strip()]
    
    @model_validator(mode="after")
    def auto_select_pedagogical_model(self) -> "CrawlRequest":
        """Select appropriate pedagogical model based on subject."""
        if self.pedagogical_model is None:
            model_mapping = {
                Subject.PHYSICS: PedagogicalModel.FIVE_E,
                Subject.CHEMISTRY: PedagogicalModel.FIVE_E,
                Subject.MATHEMATICS: PedagogicalModel.LES,
                Subject.BIOLOGY: PedagogicalModel.FIVE_E,
                Subject.HISTORY: PedagogicalModel.NARRATIVE,
                Subject.SOCIAL_STUDIES: PedagogicalModel.NARRATIVE,
                Subject.ENGLISH: PedagogicalModel.RULE_EXAMPLE,
            }
            object.__setattr__(
                self, 
                "pedagogical_model", 
                model_mapping.get(self.subject, PedagogicalModel.FIVE_E)
            )
        return self
    
    @model_validator(mode="after")
    def auto_generate_source_priorities(self) -> "CrawlRequest":
        """Generate default source priorities if not provided."""
        if not self.source_priorities:
            # Default priorities based on general effectiveness
            default_priorities = [
                SourcePriority(source=SourceType.KHAN_ACADEMY, priority=1),
                SourcePriority(source=SourceType.BYJUS, priority=2),
                SourcePriority(source=SourceType.VEDANTU, priority=3),
                SourcePriority(source=SourceType.YOUTUBE, priority=4),
                SourcePriority(source=SourceType.GOOGLE_SEARCH, priority=5),
            ]
            # Filter to only enabled sources
            filtered = [
                p for p in default_priorities 
                if p.source in self.enabled_sources or 
                   (p.source == SourceType.GOOGLE_SEARCH and self.include_google_fallback)
            ]
            object.__setattr__(self, "source_priorities", filtered)
        return self
    
    def get_search_query(self) -> str:
        """Generate optimized search query from request."""
        parts = [f"grade {self.grade}", self.subject.value, self.topic]
        if self.subtopics:
            parts.append(self.subtopics[0])  # Include first subtopic
        return " ".join(parts)
    
    def get_source_by_priority(self) -> list[SourceType]:
        """Return sources ordered by priority (highest first)."""
        return [
            p.source 
            for p in sorted(self.source_priorities, key=lambda x: x.priority)
        ]


# Type aliases for convenience
CrawlRequestDict = dict[str, any]
SourcePriorityList = list[SourcePriority]


# Example usage and testing
if __name__ == "__main__":
    # Create a sample request
    request = CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work, Energy and Power",
        subtopics=["Work Done", "Kinetic Energy", "Potential Energy"],
        youtube_links=[
            YouTubeLink(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ],
        output_type=OutputType.STUDY_MATERIAL
    )
    
    print(f"Request ID: {request.request_id}")
    print(f"Search Query: {request.get_search_query()}")
    print(f"Pedagogical Model: {request.pedagogical_model}")
    print(f"Source Priority: {request.get_source_by_priority()}")
    print(f"\nFull Request JSON:\n{request.model_dump_json(indent=2)}")
