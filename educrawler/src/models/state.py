"""
EduCrawler State Machine Models
===============================

Pydantic models for crawler state management:
- CrawlState: Current execution state
- ExecutionPlan: Planned actions
- StateTransition: State change records
- CrawlResult: Final result with metadata

Usage:
    from models.state import CrawlState, StateMachine
    
    machine = StateMachine(initial_state=CrawlStateEnum.IDLE)
    machine.transition(CrawlStateEnum.PLANNING)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


# Import from other models
from .request import CrawlRequest, SourceType, SourcePriority, QueryType
from .content import ExtractedContent, ProcessedContent, AggregatedContent
from .output import OutputDocument


class CrawlStateEnum(str, Enum):
    """Crawler state machine states."""
    
    IDLE = "idle"                       # Waiting for request
    PLANNING = "planning"               # Analyzing query, prioritizing sources
    CRAWLING = "crawling"               # Extracting content from sources
    PROCESSING = "processing"           # Cleaning and normalizing content
    AGGREGATING = "aggregating"         # Merging and deduplicating
    COMPILING = "compiling"             # NotebookLM compilation
    RENDERING = "rendering"             # Generating output format
    COMPLETED = "completed"             # Successfully finished
    ERROR = "error"                     # Failed with error
    CANCELLED = "cancelled"             # User cancelled


class SourceTaskStatus(str, Enum):
    """Status of individual source extraction task."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class SourceTask(BaseModel):
    """Individual source extraction task."""
    
    id: UUID = Field(default_factory=uuid4)
    source: SourceType
    priority: int = Field(ge=1, le=10)
    
    # URLs to crawl for this source
    urls: list[str] = Field(default_factory=list)
    
    # Configuration
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    required: bool = Field(default=False)
    
    # Status tracking
    status: SourceTaskStatus = Field(default=SourceTaskStatus.PENDING)
    retry_count: int = Field(default=0)
    
    # Timing
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Results
    extracted_content: Optional[ExtractedContent] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Check if task is in terminal state."""
        return self.status in [
            SourceTaskStatus.COMPLETED,
            SourceTaskStatus.FAILED,
            SourceTaskStatus.SKIPPED,
            SourceTaskStatus.TIMEOUT,
        ]
    
    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = SourceTaskStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, content: ExtractedContent) -> None:
        """Mark task as completed with content."""
        self.status = SourceTaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.extracted_content = content
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = SourceTaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries


class ExecutionPlan(BaseModel):
    """
    Planned execution strategy for a crawl request.
    
    Created by the Planner component based on query analysis.
    """
    
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    
    # Query analysis
    query_type: QueryType
    search_query: str
    
    # Source tasks in priority order
    tasks: list[SourceTask] = Field(default_factory=list)
    
    # Execution configuration
    max_parallel: int = Field(default=3)
    global_timeout_seconds: int = Field(default=300)
    
    # Planning metadata
    planned_at: datetime = Field(default_factory=datetime.utcnow)
    planning_duration_seconds: float = Field(default=0)
    
    @computed_field
    @property
    def total_tasks(self) -> int:
        return len(self.tasks)
    
    @computed_field
    @property
    def pending_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status == SourceTaskStatus.PENDING)
    
    @computed_field
    @property
    def completed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status == SourceTaskStatus.COMPLETED)
    
    @computed_field
    @property
    def failed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status == SourceTaskStatus.FAILED)
    
    def get_next_tasks(self, count: int = 1) -> list[SourceTask]:
        """Get next pending tasks by priority."""
        pending = [t for t in self.tasks if t.status == SourceTaskStatus.PENDING]
        pending.sort(key=lambda t: t.priority)
        return pending[:count]
    
    def get_successful_content(self) -> list[ExtractedContent]:
        """Get all successfully extracted content."""
        return [
            t.extracted_content
            for t in self.tasks
            if t.status == SourceTaskStatus.COMPLETED and t.extracted_content
        ]
    
    def all_tasks_complete(self) -> bool:
        """Check if all tasks are in terminal state."""
        return all(t.is_terminal for t in self.tasks)
    
    @classmethod
    def from_request(
        cls,
        request: CrawlRequest,
        query_type: QueryType
    ) -> "ExecutionPlan":
        """
        Create execution plan from crawl request.
        
        Args:
            request: The crawl request
            query_type: Classified query type
            
        Returns:
            ExecutionPlan with source tasks
        """
        tasks = []
        
        # Create tasks from source priorities
        for priority in request.source_priorities:
            if priority.source not in request.enabled_sources:
                continue
            
            # Generate URLs for this source
            urls = cls._generate_source_urls(
                source=priority.source,
                request=request
            )
            
            task = SourceTask(
                source=priority.source,
                priority=priority.priority,
                urls=urls,
                timeout_seconds=priority.timeout_seconds,
                max_retries=priority.max_retries,
                required=priority.required,
            )
            tasks.append(task)
        
        # Add YouTube tasks for provided links
        if request.youtube_links:
            youtube_task = SourceTask(
                source=SourceType.YOUTUBE,
                priority=4,
                urls=[link.url for link in request.youtube_links],
                timeout_seconds=30,
                max_retries=2,
                required=False,
            )
            # Check if YouTube task already exists
            existing = [t for t in tasks if t.source == SourceType.YOUTUBE]
            if existing:
                existing[0].urls.extend(youtube_task.urls)
            else:
                tasks.append(youtube_task)
        
        return cls(
            request_id=request.request_id,
            query_type=query_type,
            search_query=request.get_search_query(),
            tasks=tasks,
            max_parallel=request.config.max_parallel_sources,
            global_timeout_seconds=request.config.global_timeout_seconds,
        )
    
    @staticmethod
    def _generate_source_urls(source: SourceType, request: CrawlRequest) -> list[str]:
        """Generate URLs for a source based on request."""
        topic_slug = request.topic.lower().replace(" ", "-").replace(",", "")
        subject = request.subject.value
        grade = request.grade
        
        url_templates = {
            SourceType.KHAN_ACADEMY: [
                f"https://www.khanacademy.org/science/{subject}/{topic_slug}",
                f"https://www.khanacademy.org/science/{subject}/work-energy/{topic_slug}",
            ],
            SourceType.BYJUS: [
                f"https://byjus.com/{subject}/{topic_slug}/",
                f"https://byjus.com/cbse-notes/class-{grade}-{subject}-{topic_slug}/",
            ],
            SourceType.VEDANTU: [
                f"https://www.vedantu.com/{subject}/{topic_slug}",
                f"https://www.vedantu.com/ncert-solutions/class-{grade}-science-chapter-{topic_slug}",
            ],
            SourceType.GOOGLE_SEARCH: [
                f"https://www.google.com/search?q=grade+{grade}+{subject}+{topic_slug}+explanation",
            ],
        }
        
        return url_templates.get(source, [])


class StateTransition(BaseModel):
    """Record of a state transition."""
    
    id: UUID = Field(default_factory=uuid4)
    from_state: CrawlStateEnum
    to_state: CrawlStateEnum
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trigger: str = Field(default="", description="What triggered the transition")
    metadata: dict = Field(default_factory=dict)


class CrawlState(BaseModel):
    """
    Current state of a crawl operation.
    
    Tracks progress through the state machine and holds
    intermediate results.
    """
    
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    
    # Current state
    current_state: CrawlStateEnum = Field(default=CrawlStateEnum.IDLE)
    
    # State history
    transitions: list[StateTransition] = Field(default_factory=list)
    
    # Execution plan (set during PLANNING)
    execution_plan: Optional[ExecutionPlan] = Field(default=None)
    
    # Extracted content (set during CRAWLING)
    extracted_contents: list[ExtractedContent] = Field(default_factory=list)
    
    # Processed content (set during PROCESSING)
    processed_contents: list[ProcessedContent] = Field(default_factory=list)
    
    # Aggregated content (set during AGGREGATING)
    aggregated_content: Optional[AggregatedContent] = Field(default=None)
    
    # NotebookLM session (set during COMPILING)
    notebooklm_notebook_id: Optional[str] = Field(default=None)
    notebooklm_compiled_content: Optional[str] = Field(default=None)
    
    # Final output (set during RENDERING)
    output_document: Optional[OutputDocument] = Field(default=None)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Error tracking
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Total duration of crawl operation."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.utcnow() - self.started_at).total_seconds()
    
    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self.current_state in [
            CrawlStateEnum.COMPLETED,
            CrawlStateEnum.ERROR,
            CrawlStateEnum.CANCELLED,
        ]
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Estimated progress percentage."""
        state_progress = {
            CrawlStateEnum.IDLE: 0,
            CrawlStateEnum.PLANNING: 10,
            CrawlStateEnum.CRAWLING: 40,
            CrawlStateEnum.PROCESSING: 60,
            CrawlStateEnum.AGGREGATING: 70,
            CrawlStateEnum.COMPILING: 85,
            CrawlStateEnum.RENDERING: 95,
            CrawlStateEnum.COMPLETED: 100,
            CrawlStateEnum.ERROR: 100,
            CrawlStateEnum.CANCELLED: 100,
        }
        return state_progress.get(self.current_state, 0)
    
    def transition_to(
        self,
        new_state: CrawlStateEnum,
        trigger: str = "",
        metadata: dict = None
    ) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state
            trigger: What triggered this transition
            metadata: Additional metadata
        """
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            metadata=metadata or {}
        )
        self.transitions.append(transition)
        self.current_state = new_state
        
        if new_state in [CrawlStateEnum.COMPLETED, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED]:
            self.completed_at = datetime.utcnow()
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(f"[{datetime.utcnow().isoformat()}] {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"[{datetime.utcnow().isoformat()}] {warning}")


class CrawlResult(BaseModel):
    """
    Final result of a crawl operation.
    
    Returned to the caller with all relevant information.
    """
    
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    
    # Success status
    success: bool
    
    # Output (if successful)
    output: Optional[OutputDocument] = Field(default=None)
    
    # Metadata
    sources_attempted: list[SourceType] = Field(default_factory=list)
    sources_succeeded: list[SourceType] = Field(default_factory=list)
    sources_failed: list[SourceType] = Field(default_factory=list)
    
    # Timing
    total_duration_seconds: float
    planning_duration_seconds: float = Field(default=0)
    crawling_duration_seconds: float = Field(default=0)
    processing_duration_seconds: float = Field(default=0)
    compilation_duration_seconds: float = Field(default=0)
    rendering_duration_seconds: float = Field(default=0)
    
    # Quality metrics
    content_quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    coverage_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Errors and warnings
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    # State history (for debugging)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    
    @classmethod
    def from_state(cls, state: CrawlState) -> "CrawlResult":
        """Create result from final crawl state."""
        # Determine success
        success = (
            state.current_state == CrawlStateEnum.COMPLETED and
            state.output_document is not None
        )
        
        # Calculate phase durations from transitions
        phase_durations = cls._calculate_phase_durations(state.transitions)
        
        # Gather source stats
        sources_attempted = []
        sources_succeeded = []
        sources_failed = []
        
        if state.execution_plan:
            for task in state.execution_plan.tasks:
                sources_attempted.append(task.source)
                if task.status == SourceTaskStatus.COMPLETED:
                    sources_succeeded.append(task.source)
                elif task.status in [SourceTaskStatus.FAILED, SourceTaskStatus.TIMEOUT]:
                    sources_failed.append(task.source)
        
        return cls(
            request_id=state.request_id,
            success=success,
            output=state.output_document,
            sources_attempted=sources_attempted,
            sources_succeeded=sources_succeeded,
            sources_failed=sources_failed,
            total_duration_seconds=state.duration_seconds or 0,
            planning_duration_seconds=phase_durations.get("planning", 0),
            crawling_duration_seconds=phase_durations.get("crawling", 0),
            processing_duration_seconds=phase_durations.get("processing", 0),
            compilation_duration_seconds=phase_durations.get("compiling", 0),
            rendering_duration_seconds=phase_durations.get("rendering", 0),
            content_quality_score=(
                state.aggregated_content.overall_quality_score
                if state.aggregated_content else 0.0
            ),
            coverage_score=(
                state.aggregated_content.coverage_score
                if state.aggregated_content else 0.0
            ),
            errors=state.errors,
            warnings=state.warnings,
            state_transitions=state.transitions,
        )
    
    @staticmethod
    def _calculate_phase_durations(transitions: list[StateTransition]) -> dict[str, float]:
        """Calculate duration of each phase from transitions."""
        durations = {}
        
        for i, transition in enumerate(transitions):
            phase_name = transition.from_state.value
            
            if i + 1 < len(transitions):
                next_transition = transitions[i + 1]
                duration = (next_transition.timestamp - transition.timestamp).total_seconds()
                durations[phase_name] = duration
        
        return durations


# =============================================================================
# STATE MACHINE IMPLEMENTATION
# =============================================================================

# Valid state transitions
VALID_TRANSITIONS: dict[CrawlStateEnum, list[CrawlStateEnum]] = {
    CrawlStateEnum.IDLE: [CrawlStateEnum.PLANNING, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.PLANNING: [CrawlStateEnum.CRAWLING, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.CRAWLING: [CrawlStateEnum.PROCESSING, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.PROCESSING: [CrawlStateEnum.AGGREGATING, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.AGGREGATING: [CrawlStateEnum.COMPILING, CrawlStateEnum.RENDERING, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.COMPILING: [CrawlStateEnum.RENDERING, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.RENDERING: [CrawlStateEnum.COMPLETED, CrawlStateEnum.ERROR, CrawlStateEnum.CANCELLED],
    CrawlStateEnum.COMPLETED: [],  # Terminal state
    CrawlStateEnum.ERROR: [],       # Terminal state
    CrawlStateEnum.CANCELLED: [],   # Terminal state
}


class StateMachine:
    """
    State machine for managing crawl operations.
    
    Usage:
        machine = StateMachine(request_id=uuid4())
        await machine.run(request)
    """
    
    def __init__(self, request_id: UUID):
        self.state = CrawlState(request_id=request_id)
        self._handlers: dict[CrawlStateEnum, Callable] = {}
    
    @property
    def current_state(self) -> CrawlStateEnum:
        return self.state.current_state
    
    def can_transition_to(self, target: CrawlStateEnum) -> bool:
        """Check if transition to target state is valid."""
        return target in VALID_TRANSITIONS.get(self.current_state, [])
    
    def transition(self, target: CrawlStateEnum, trigger: str = "", metadata: dict = None) -> bool:
        """
        Attempt to transition to target state.
        
        Args:
            target: Target state
            trigger: What triggered this transition
            metadata: Additional metadata
            
        Returns:
            True if transition succeeded, False otherwise
        """
        if not self.can_transition_to(target):
            self.state.add_error(
                f"Invalid transition from {self.current_state.value} to {target.value}"
            )
            return False
        
        self.state.transition_to(target, trigger, metadata)
        return True
    
    def register_handler(
        self,
        state: CrawlStateEnum,
        handler: Callable
    ) -> None:
        """Register a handler for a state."""
        self._handlers[state] = handler
    
    async def execute_current_state(self) -> None:
        """Execute handler for current state."""
        handler = self._handlers.get(self.current_state)
        if handler:
            await handler(self.state)
    
    def get_result(self) -> CrawlResult:
        """Get final result."""
        return CrawlResult.from_state(self.state)


# Example usage
if __name__ == "__main__":
    from .request import CrawlRequest, Subject, OutputType
    
    # Create a request
    request = CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work Done",
        output_type=OutputType.STUDY_MATERIAL
    )
    
    # Create execution plan
    plan = ExecutionPlan.from_request(request, QueryType.CONCEPTUAL)
    
    print(f"Execution Plan ID: {plan.id}")
    print(f"Query Type: {plan.query_type}")
    print(f"Search Query: {plan.search_query}")
    print(f"Total Tasks: {plan.total_tasks}")
    print("\nTasks:")
    for task in plan.tasks:
        print(f"  - {task.source.value} (priority: {task.priority})")
        for url in task.urls[:2]:  # Show first 2 URLs
            print(f"    • {url}")
