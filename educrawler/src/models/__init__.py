"""
EduCrawler Data Models
======================

Comprehensive Pydantic models for the educational content crawler.

Modules:
- request: CrawlRequest, CrawlConfig, SourcePriority
- content: ExtractedContent, ProcessedContent, AggregatedContent
- output: StudyMaterial, Questionnaire, Handout
- state: CrawlState, ExecutionPlan, CrawlResult

Usage:
    from models import CrawlRequest, Subject, OutputType
    
    request = CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work Done"
    )
"""

# Request models
from .request import (
    Subject,
    OutputType,
    SourceType,
    QueryType,
    PedagogicalModel,
    SourcePriority,
    CrawlConfig,
    YouTubeLink,
    CrawlRequest,
)

# Content models
from .content import (
    ContentType,
    DifficultyLevel,
    Formula,
    Example,
    Exercise,
    Concept,
    Diagram,
    VideoTranscript,
    ExtractionMetadata,
    ExtractedContent,
    ProcessedContent,
    ContentChunk,
    AggregatedContent,
)

# Output models
from .output import (
    QuestionTier,
    QuestionType,
    MCQOption,
    BaseQuestion,
    MCQQuestion,
    FillBlankQuestion,
    TrueFalseQuestion,
    AssertionReasoningQuestion,
    CalculationQuestion,
    CaseStudyQuestion,
    DetailedQuestion,
    Question,
    ConceptBox,
    StudyMaterialPage,
    StudyMaterial,
    QuestionnaireSection,
    Questionnaire,
    FormulaTableEntry,
    VisualMapNode,
    ProTip,
    Handout,
    OutputDocument,
)

# State models
from .state import (
    CrawlStateEnum,
    SourceTaskStatus,
    SourceTask,
    ExecutionPlan,
    StateTransition,
    CrawlState,
    CrawlResult,
    StateMachine,
    VALID_TRANSITIONS,
)

__all__ = [
    # Request
    "Subject",
    "OutputType",
    "SourceType",
    "QueryType",
    "PedagogicalModel",
    "SourcePriority",
    "CrawlConfig",
    "YouTubeLink",
    "CrawlRequest",
    # Content
    "ContentType",
    "DifficultyLevel",
    "Formula",
    "Example",
    "Exercise",
    "Concept",
    "Diagram",
    "VideoTranscript",
    "ExtractionMetadata",
    "ExtractedContent",
    "ProcessedContent",
    "ContentChunk",
    "AggregatedContent",
    # Output
    "QuestionTier",
    "QuestionType",
    "MCQOption",
    "BaseQuestion",
    "MCQQuestion",
    "FillBlankQuestion",
    "TrueFalseQuestion",
    "AssertionReasoningQuestion",
    "CalculationQuestion",
    "CaseStudyQuestion",
    "DetailedQuestion",
    "Question",
    "ConceptBox",
    "StudyMaterialPage",
    "StudyMaterial",
    "QuestionnaireSection",
    "Questionnaire",
    "FormulaTableEntry",
    "VisualMapNode",
    "ProTip",
    "Handout",
    "OutputDocument",
    # State
    "CrawlStateEnum",
    "SourceTaskStatus",
    "SourceTask",
    "ExecutionPlan",
    "StateTransition",
    "CrawlState",
    "CrawlResult",
    "StateMachine",
    "VALID_TRANSITIONS",
]
