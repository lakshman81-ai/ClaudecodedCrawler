"""
EduCrawler Content Models
=========================

Pydantic models for extracted, processed, and aggregated educational content.

These models represent content at various stages of the pipeline:
1. ExtractedContent - Raw content from a single source
2. ProcessedContent - Cleaned and normalized content
3. AggregatedContent - Merged content from multiple sources

Usage:
    from models.content import ExtractedContent, Concept, Formula
    
    content = ExtractedContent(
        source=SourceType.KHAN_ACADEMY,
        url="https://khanacademy.org/...",
        concepts=[Concept(title="Work", definition="...")],
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, computed_field


# Import from request models
from .request import SourceType, Subject


class ContentType(str, Enum):
    """Classification of content type for extraction."""
    CONCEPT = "concept"
    FORMULA = "formula"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    DEFINITION = "definition"
    DIAGRAM = "diagram"
    VIDEO_TRANSCRIPT = "video_transcript"
    TABLE = "table"
    LIST = "list"


class DifficultyLevel(str, Enum):
    """Difficulty classification for exercises and examples."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Formula(BaseModel):
    """Mathematical or scientific formula with LaTeX representation."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    latex: str = Field(description="LaTeX representation")
    unicode: str = Field(default="", description="Unicode fallback representation")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variable definitions {symbol: description}"
    )
    units: Optional[str] = Field(default=None, description="SI units for the result")
    context: Optional[str] = Field(default=None, description="When to use this formula")
    
    @field_validator("latex")
    @classmethod
    def validate_latex(cls, v: str) -> str:
        """Basic LaTeX validation."""
        # Check for unmatched braces
        if v.count("{") != v.count("}"):
            raise ValueError("Unmatched braces in LaTeX formula")
        return v.strip()
    
    def to_unicode(self) -> str:
        """Convert LaTeX to Unicode approximation."""
        if self.unicode:
            return self.unicode
        # Basic conversion (extend as needed)
        replacements = {
            "\\times": "×",
            "\\div": "÷",
            "\\pm": "±",
            "\\leq": "≤",
            "\\geq": "≥",
            "\\neq": "≠",
            "\\approx": "≈",
            "\\infty": "∞",
            "\\sqrt": "√",
            "\\pi": "π",
            "\\theta": "θ",
            "\\alpha": "α",
            "\\beta": "β",
            "\\gamma": "γ",
            "\\delta": "δ",
            "\\Delta": "Δ",
            "\\lambda": "λ",
            "\\mu": "μ",
            "\\Omega": "Ω",
            "^2": "²",
            "^3": "³",
            "_0": "₀",
            "_1": "₁",
            "_2": "₂",
        }
        result = self.latex
        for latex_sym, unicode_sym in replacements.items():
            result = result.replace(latex_sym, unicode_sym)
        return result


class Example(BaseModel):
    """Worked example with step-by-step solution."""
    
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=300)
    problem_statement: str
    given_data: dict[str, str] = Field(
        default_factory=dict,
        description="Known values {variable: value with unit}"
    )
    find: list[str] = Field(default_factory=list, description="What to find")
    solution_steps: list[str] = Field(min_length=1)
    final_answer: str
    formulas_used: list[UUID] = Field(
        default_factory=list,
        description="References to Formula IDs"
    )
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    
    @computed_field
    @property
    def step_count(self) -> int:
        return len(self.solution_steps)


class Exercise(BaseModel):
    """Practice problem for student assessment."""
    
    id: UUID = Field(default_factory=uuid4)
    problem_statement: str
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    hints: list[str] = Field(default_factory=list)
    answer: Optional[str] = Field(default=None)
    detailed_solution: Optional[str] = Field(default=None)
    related_concepts: list[UUID] = Field(
        default_factory=list,
        description="References to Concept IDs"
    )
    marks: int = Field(default=1, ge=1, le=10)


class Concept(BaseModel):
    """Core concept or topic explanation."""
    
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=300)
    definition: str = Field(min_length=10)
    explanation: str = Field(default="")
    key_points: list[str] = Field(default_factory=list)
    
    # Related content
    formulas: list[Formula] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    
    # Metadata
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Concepts that should be understood first"
    )
    related_topics: list[str] = Field(default_factory=list)
    
    # Educational boxes (per validation checklist requirements)
    concept_helper: Optional[str] = Field(
        default=None,
        description="💡 Concept Helper box content"
    )
    common_misunderstanding: Optional[str] = Field(
        default=None,
        description="⚠️ Common Misunderstanding box content"
    )
    real_world_application: Optional[str] = Field(
        default=None,
        description="🌍 Real-World Application box content"
    )
    did_you_know: Optional[str] = Field(
        default=None,
        description="🔍 Did You Know box content"
    )
    
    @computed_field
    @property
    def has_required_boxes(self) -> bool:
        """Check if concept has minimum required educational boxes."""
        boxes = [
            self.concept_helper,
            self.common_misunderstanding,
            self.real_world_application,
            self.did_you_know
        ]
        return sum(1 for b in boxes if b) >= 2


class Diagram(BaseModel):
    """Visual diagram or image reference."""
    
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    source_url: Optional[str] = Field(default=None)
    local_path: Optional[str] = Field(default=None)
    alt_text: str = Field(description="Accessibility alt text")
    caption: str = Field(max_length=100, description="≤15 word caption")
    width_px: Optional[int] = Field(default=None, ge=100)
    height_px: Optional[int] = Field(default=None, ge=100)
    
    @field_validator("caption")
    @classmethod
    def validate_caption_length(cls, v: str) -> str:
        """Ensure caption is ≤15 words."""
        word_count = len(v.split())
        if word_count > 15:
            raise ValueError(f"Caption must be ≤15 words, got {word_count}")
        return v


class VideoTranscript(BaseModel):
    """Extracted YouTube video transcript."""
    
    id: UUID = Field(default_factory=uuid4)
    video_id: str = Field(min_length=11, max_length=11)
    video_url: str
    title: str
    channel: Optional[str] = Field(default=None)
    duration_seconds: Optional[int] = Field(default=None)
    
    transcript_text: str
    is_auto_generated: bool = Field(default=False)
    language: str = Field(default="en")
    confidence_score: float = Field(default=0.95, ge=0.0, le=1.0)
    
    # Extracted segments
    key_timestamps: list[dict[str, any]] = Field(
        default_factory=list,
        description="[{time_seconds: int, text: str, topic: str}]"
    )
    
    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.transcript_text.split())


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_duration_seconds: float = Field(ge=0)
    page_load_time_seconds: Optional[float] = Field(default=None)
    
    # Quality metrics
    content_completeness_score: float = Field(ge=0.0, le=1.0)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    
    # Technical details
    http_status_code: Optional[int] = Field(default=None)
    js_rendering_required: bool = Field(default=False)
    retry_count: int = Field(default=0, ge=0)
    fallback_strategy_used: Optional[str] = Field(default=None)
    
    # Errors
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractedContent(BaseModel):
    """
    Raw content extracted from a single source.
    
    This is the output of a SourceHandler's extract() method.
    Contains structured educational content ready for processing.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Source identification
    source: SourceType
    url: str
    page_title: str = Field(default="")
    
    # Educational content
    subject: Optional[Subject] = Field(default=None)
    topic: str = Field(default="")
    
    # Structured content
    concepts: list[Concept] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    exercises: list[Exercise] = Field(default_factory=list)
    diagrams: list[Diagram] = Field(default_factory=list)
    video_transcripts: list[VideoTranscript] = Field(default_factory=list)
    
    # Raw content (for fallback processing)
    raw_html: Optional[str] = Field(default=None, repr=False)
    raw_markdown: Optional[str] = Field(default=None, repr=False)
    raw_text: Optional[str] = Field(default=None, repr=False)
    
    # Metadata
    metadata: ExtractionMetadata = Field(default_factory=lambda: ExtractionMetadata(
        extraction_duration_seconds=0,
        content_completeness_score=0,
        extraction_confidence=0
    ))
    
    @computed_field
    @property
    def content_count(self) -> dict[str, int]:
        """Count of each content type."""
        return {
            "concepts": len(self.concepts),
            "formulas": len(self.formulas),
            "examples": len(self.examples),
            "exercises": len(self.exercises),
            "diagrams": len(self.diagrams),
            "video_transcripts": len(self.video_transcripts),
        }
    
    @computed_field
    @property
    def total_content_items(self) -> int:
        """Total number of content items extracted."""
        return sum(self.content_count.values())
    
    @computed_field
    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score (0-1).
        
        Factors:
        - Content completeness (40%)
        - Extraction confidence (30%)
        - Content diversity (30%)
        """
        # Content completeness from metadata
        completeness = self.metadata.content_completeness_score
        
        # Extraction confidence from metadata
        confidence = self.metadata.extraction_confidence
        
        # Content diversity (having multiple content types)
        content_types_present = sum(1 for v in self.content_count.values() if v > 0)
        diversity = min(content_types_present / 4, 1.0)  # Max out at 4 types
        
        return (completeness * 0.4) + (confidence * 0.3) + (diversity * 0.3)
    
    def is_valid(self, min_quality: float = 0.6) -> bool:
        """Check if content meets minimum quality threshold."""
        return self.quality_score >= min_quality and self.total_content_items > 0


class ProcessedContent(BaseModel):
    """
    Cleaned and normalized content from a single source.
    
    Processing includes:
    - HTML to Markdown conversion
    - LaTeX/MathML preservation
    - Text normalization
    - Quality scoring
    """
    
    id: UUID = Field(default_factory=uuid4)
    source_content_id: UUID = Field(description="Reference to ExtractedContent.id")
    
    # Source info (copied from ExtractedContent)
    source: SourceType
    url: str
    
    # Processed content
    concepts: list[Concept] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    exercises: list[Exercise] = Field(default_factory=list)
    diagrams: list[Diagram] = Field(default_factory=list)
    video_transcripts: list[VideoTranscript] = Field(default_factory=list)
    
    # Processing metadata
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_duration_seconds: float = Field(ge=0)
    
    # Quality metrics
    original_quality_score: float = Field(ge=0.0, le=1.0)
    processed_quality_score: float = Field(ge=0.0, le=1.0)
    
    # Processing notes
    transformations_applied: list[str] = Field(default_factory=list)
    content_removed: list[str] = Field(
        default_factory=list,
        description="Descriptions of content removed during processing"
    )


class ContentChunk(BaseModel):
    """
    A chunk of content for deduplication comparison.
    Used internally by the aggregator.
    """
    
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    source_type: SourceType
    content_type: ContentType
    text: str
    embedding: Optional[list[float]] = Field(default=None, repr=False)
    minhash_signature: Optional[list[int]] = Field(default=None, repr=False)


class AggregatedContent(BaseModel):
    """
    Merged content from multiple sources with deduplication.
    
    This is the final content model before NotebookLM compilation
    or direct rendering.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Request context
    topic: str
    subject: Subject
    grade: int
    
    # Source tracking
    sources_used: list[SourceType] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    
    # Merged content (deduplicated)
    concepts: list[Concept] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    exercises: list[Exercise] = Field(default_factory=list)
    diagrams: list[Diagram] = Field(default_factory=list)
    video_transcripts: list[VideoTranscript] = Field(default_factory=list)
    
    # Aggregation metadata
    aggregated_at: datetime = Field(default_factory=datetime.utcnow)
    total_sources_attempted: int = Field(default=0)
    successful_sources: int = Field(default=0)
    
    # Deduplication stats
    original_content_count: int = Field(default=0)
    deduplicated_content_count: int = Field(default=0)
    duplicates_removed: int = Field(default=0)
    
    # Quality
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(
        ge=0.0, le=1.0,
        description="How well subtopics are covered"
    )
    
    @computed_field
    @property
    def deduplication_ratio(self) -> float:
        """Ratio of content removed as duplicates."""
        if self.original_content_count == 0:
            return 0.0
        return self.duplicates_removed / self.original_content_count
    
    def get_formulas_by_concept(self, concept_id: UUID) -> list[Formula]:
        """Get formulas referenced by a specific concept."""
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept.formulas
        return []
    
    def to_notebooklm_text(self) -> str:
        """
        Convert aggregated content to text format for NotebookLM.
        
        Returns a markdown-formatted string suitable for pasting
        as a NotebookLM source.
        """
        sections = []
        
        # Title
        sections.append(f"# {self.topic}\n")
        sections.append(f"**Subject:** {self.subject.value.title()}")
        sections.append(f"**Grade:** {self.grade}\n")
        
        # Concepts
        if self.concepts:
            sections.append("## Key Concepts\n")
            for concept in self.concepts:
                sections.append(f"### {concept.title}")
                sections.append(f"{concept.definition}\n")
                if concept.explanation:
                    sections.append(concept.explanation)
                if concept.key_points:
                    sections.append("\n**Key Points:**")
                    for point in concept.key_points:
                        sections.append(f"- {point}")
                sections.append("")
        
        # Formulas
        if self.formulas:
            sections.append("## Formulas\n")
            for formula in self.formulas:
                sections.append(f"**{formula.name}:** {formula.to_unicode()}")
                if formula.variables:
                    sections.append("Where:")
                    for var, desc in formula.variables.items():
                        sections.append(f"  - {var} = {desc}")
                sections.append("")
        
        # Examples
        if self.examples:
            sections.append("## Worked Examples\n")
            for i, example in enumerate(self.examples, 1):
                sections.append(f"### Example {i}: {example.title}")
                sections.append(f"**Problem:** {example.problem_statement}")
                sections.append("**Solution:**")
                for j, step in enumerate(example.solution_steps, 1):
                    sections.append(f"{j}. {step}")
                sections.append(f"**Answer:** {example.final_answer}\n")
        
        return "\n".join(sections)


# Example usage
if __name__ == "__main__":
    # Create sample content
    formula = Formula(
        name="Work Formula",
        latex="W = F \\times d \\times \\cos{\\theta}",
        variables={
            "W": "Work done (Joules)",
            "F": "Force applied (Newtons)",
            "d": "Displacement (meters)",
            "θ": "Angle between force and displacement"
        },
        units="Joules (J)"
    )
    
    concept = Concept(
        title="Work Done",
        definition="Work is done when a force causes displacement of an object.",
        explanation="In physics, work has a specific meaning...",
        key_points=[
            "Work = Force × Displacement × cos(θ)",
            "Work is a scalar quantity",
            "SI unit is Joule (J)"
        ],
        formulas=[formula],
        concept_helper="Think of work as energy transfer through force.",
        common_misunderstanding="Holding a heavy object stationary is NOT doing work in physics.",
    )
    
    content = ExtractedContent(
        source=SourceType.KHAN_ACADEMY,
        url="https://khanacademy.org/science/physics/work-energy",
        page_title="Work and Energy",
        topic="Work Done",
        concepts=[concept],
        formulas=[formula],
        metadata=ExtractionMetadata(
            extraction_duration_seconds=5.2,
            content_completeness_score=0.85,
            extraction_confidence=0.92
        )
    )
    
    print(f"Content ID: {content.id}")
    print(f"Quality Score: {content.quality_score:.2f}")
    print(f"Content Count: {content.content_count}")
    print(f"Is Valid: {content.is_valid()}")
