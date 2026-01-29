# EduCrawler: Error Handling and Testing Specifications
## Part 2 of Technical Specification

---

## 9. Error Handling Specifications

### 9.1 Error Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR CLASSIFICATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RECOVERABLE ERRORS (Auto-retry)                                            │
│  ├── Network timeout                                                        │
│  ├── HTTP 429 (Too Many Requests)                                          │
│  ├── HTTP 500, 502, 503, 504 (Server errors)                               │
│  ├── DNS resolution failure                                                 │
│  ├── Connection reset                                                       │
│  └── Temporary Cloudflare challenge                                         │
│                                                                              │
│  RECOVERABLE ERRORS (Fallback to alternative)                               │
│  ├── Content extraction failure                                            │
│  ├── MathJax rendering timeout                                             │
│  ├── Empty content extracted                                               │
│  └── Quality score below threshold                                          │
│                                                                              │
│  NON-RECOVERABLE ERRORS (Fail task)                                        │
│  ├── HTTP 404 (Not Found)                                                  │
│  ├── HTTP 403 (Forbidden) - persistent                                     │
│  ├── Invalid URL format                                                     │
│  ├── Authentication required                                                │
│  └── Content behind paywall                                                 │
│                                                                              │
│  CRITICAL ERRORS (Abort workflow)                                          │
│  ├── All sources failed                                                    │
│  ├── NotebookLM authentication expired                                      │
│  ├── Global timeout exceeded                                                │
│  └── Out of memory                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Retry Strategy

```python
# Pseudocode for retry logic

class RetryStrategy:
    """
    Exponential backoff with jitter.
    
    Formula: delay = min(backoff_max, backoff_base ** attempt) + random(0, jitter)
    """
    
    BACKOFF_BASE = 2.0
    BACKOFF_MAX = 32.0
    JITTER_MAX = 0.5
    MAX_RETRIES = 3
    
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
    
    async def execute_with_retry(self, operation, *args):
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                result = await operation(*args)
                
                # Check if result indicates retry needed
                if self.should_retry(result, attempt):
                    delay = self.calculate_delay(attempt, result)
                    await asyncio.sleep(delay)
                    continue
                    
                return result
                
            except RetryableError as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.calculate_delay(attempt)
                    log.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    
            except NonRetryableError as e:
                raise  # Don't retry
        
        raise MaxRetriesExceeded(last_error)
    
    def calculate_delay(self, attempt: int, result=None) -> float:
        # Check for Retry-After header
        if result and hasattr(result, 'headers'):
            retry_after = result.headers.get('Retry-After')
            if retry_after:
                return float(retry_after)
        
        # Exponential backoff
        delay = self.BACKOFF_BASE ** attempt
        delay = min(delay, self.BACKOFF_MAX)
        
        # Add jitter
        jitter = random.uniform(0, self.JITTER_MAX)
        delay += jitter
        
        return delay
```

### 9.3 Circuit Breaker Pattern

```python
# Pseudocode for circuit breaker

class CircuitBreaker:
    """
    Prevents repeated failures by "opening" the circuit.
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT = 300  # 5 minutes
    
    def __init__(self, domain: str):
        self.domain = domain
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.RECOVERY_TIMEOUT:
                self.state = "HALF_OPEN"
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True  # Allow single test request
        
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.FAILURE_THRESHOLD:
            self.state = "OPEN"
            log.warning(f"Circuit OPEN for {self.domain}")
```

### 9.4 Error Response Format

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str  # e.g., "EXTRACTION_FAILED", "TIMEOUT"
    message: str
    source: Optional[str] = None  # Which source/handler
    url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Recovery information
    is_recoverable: bool = True
    retry_count: int = 0
    fallback_used: Optional[str] = None
    
    # Debug information (excluded from production logs)
    stack_trace: Optional[str] = None
    raw_response: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    success: bool = False
    error: ErrorDetail
    partial_content: Optional[dict] = None  # Any content extracted before failure
    suggestions: list[str] = []  # Possible recovery actions
```

---

## 10. Testing Specifications

### 10.1 Test Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST CATEGORY MATRIX                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UNIT TESTS                                                                  │
│  ├── Models (validation, serialization)                                    │
│  ├── Content processors (HTML→MD, deduplication)                           │
│  ├── Output renderers (template rendering)                                 │
│  └── Utility functions (URL parsing, rate limiting)                        │
│                                                                              │
│  INTEGRATION TESTS                                                          │
│  ├── Source handler → Content processor pipeline                           │
│  ├── Aggregator → NotebookLM integration                                   │
│  ├── End-to-end crawl workflow                                             │
│  └── State machine transitions                                              │
│                                                                              │
│  MOCK TESTS (No external dependencies)                                      │
│  ├── Mocked HTTP responses for each source                                 │
│  ├── Mocked NotebookLM DOM interactions                                    │
│  └── Mocked YouTube transcript API                                          │
│                                                                              │
│  LIVE TESTS (With external services - run sparingly)                       │
│  ├── Actual source extraction (rate limited)                               │
│  ├── NotebookLM compilation (manual verification)                          │
│  └── YouTube transcript extraction                                          │
│                                                                              │
│  OUTPUT VALIDATION TESTS                                                    │
│  ├── Study Material structure validation                                   │
│  ├── Questionnaire count validation (exactly 50)                           │
│  ├── Handout format validation                                             │
│  └── Content quality scoring                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Test Fixtures

```python
# tests/fixtures/sample_content.py

"""
Sample content fixtures for testing.
"""

import pytest
from models import (
    CrawlRequest, Subject, OutputType,
    ExtractedContent, Concept, Formula, Example,
    SourceType
)


@pytest.fixture
def sample_crawl_request():
    """Standard crawl request for testing."""
    return CrawlRequest(
        grade=8,
        subject=Subject.PHYSICS,
        topic="Work, Energy and Power",
        subtopics=["Work Done", "Kinetic Energy", "Potential Energy"],
        output_type=OutputType.STUDY_MATERIAL
    )


@pytest.fixture
def sample_formula():
    """Sample physics formula."""
    return Formula(
        name="Work Formula",
        latex="W = F \\times d \\times \\cos{\\theta}",
        unicode="W = F × d × cos(θ)",
        variables={
            "W": "Work done (Joules)",
            "F": "Force applied (Newtons)",
            "d": "Displacement (meters)",
            "θ": "Angle between force and displacement"
        },
        units="Joules (J)",
        context="Use when calculating work with force at an angle"
    )


@pytest.fixture
def sample_concept(sample_formula):
    """Sample physics concept."""
    return Concept(
        title="Work Done",
        definition="Work is done when a force causes displacement of an object in the direction of the force.",
        explanation="In physics, work has a specific technical meaning...",
        key_points=[
            "Work = Force × Displacement × cos(θ)",
            "Work is a scalar quantity",
            "SI unit is Joule (J)",
            "No work is done if displacement is zero"
        ],
        formulas=[sample_formula],
        concept_helper="Think of work as energy transfer through force application.",
        common_misunderstanding="Holding a heavy object stationary does NOT constitute work in physics, because there is no displacement.",
        real_world_application="When you push a shopping cart, you do work on it. The faster you push, the more power you exert.",
        did_you_know="James Joule discovered that 4.18 J of mechanical work produces 1 calorie of heat energy."
    )


@pytest.fixture
def sample_example():
    """Sample worked example."""
    return Example(
        title="Calculating Work Done",
        problem_statement="A person pushes a box with a force of 50 N across a floor for a distance of 10 m. The force is applied at an angle of 30° to the horizontal. Calculate the work done.",
        given_data={
            "F": "50 N",
            "d": "10 m",
            "θ": "30°"
        },
        find=["Work done (W)"],
        solution_steps=[
            "Write the formula: W = F × d × cos(θ)",
            "Substitute values: W = 50 × 10 × cos(30°)",
            "Calculate cos(30°) = 0.866",
            "W = 50 × 10 × 0.866 = 433 J"
        ],
        final_answer="433 J"
    )


@pytest.fixture
def sample_extracted_content(sample_concept, sample_formula, sample_example):
    """Sample extracted content from a source."""
    from models import ExtractionMetadata
    
    return ExtractedContent(
        source=SourceType.KHAN_ACADEMY,
        url="https://www.khanacademy.org/science/physics/work-and-energy/work-and-energy-tutorial/v/introduction-to-work",
        page_title="Introduction to Work and Energy",
        topic="Work Done",
        concepts=[sample_concept],
        formulas=[sample_formula],
        examples=[sample_example],
        metadata=ExtractionMetadata(
            extraction_duration_seconds=5.2,
            content_completeness_score=0.85,
            extraction_confidence=0.92
        )
    )


@pytest.fixture
def mock_khan_html():
    """Mock HTML response from Khan Academy."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Work and Energy | Physics | Khan Academy</title></head>
    <body>
        <div class="article-content">
            <h1 class="lesson-title">Introduction to Work and Energy</h1>
            <div class="paragraph">
                Work is defined as the transfer of energy when a force acts 
                on an object causing displacement.
            </div>
            <div class="MathJax_Display">
                <script type="math/tex">W = F \\cdot d \\cdot \\cos(\\theta)</script>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def mock_youtube_transcript():
    """Mock YouTube transcript response."""
    return [
        {"text": "Today we're going to learn about work in physics.", "start": 0.0, "duration": 3.5},
        {"text": "Work is defined as force times displacement.", "start": 3.5, "duration": 4.0},
        {"text": "The formula is W equals F times d times cosine theta.", "start": 7.5, "duration": 5.0},
        {"text": "Let's look at an example.", "start": 12.5, "duration": 2.0},
    ]
```

### 10.3 Unit Test Examples

```python
# tests/unit/test_models.py

"""
Unit tests for data models.
"""

import pytest
from pydantic import ValidationError
from models import (
    CrawlRequest, Subject, OutputType,
    Formula, Concept, MCQQuestion, QuestionTier,
    Questionnaire, QuestionnaireSection
)


class TestCrawlRequest:
    """Tests for CrawlRequest model."""
    
    def test_valid_request(self, sample_crawl_request):
        """Test valid request creation."""
        assert sample_crawl_request.grade == 8
        assert sample_crawl_request.subject == Subject.PHYSICS
        assert sample_crawl_request.topic == "Work, Energy and Power"
    
    def test_grade_validation(self):
        """Test grade must be between 1-12."""
        with pytest.raises(ValidationError):
            CrawlRequest(
                grade=15,  # Invalid
                subject=Subject.PHYSICS,
                topic="Test"
            )
    
    def test_auto_pedagogical_model(self):
        """Test automatic pedagogical model selection."""
        physics_req = CrawlRequest(
            grade=8,
            subject=Subject.PHYSICS,
            topic="Forces"
        )
        assert physics_req.pedagogical_model.value == "5e"
        
        math_req = CrawlRequest(
            grade=8,
            subject=Subject.MATHEMATICS,
            topic="Algebra"
        )
        assert math_req.pedagogical_model.value == "les"
    
    def test_search_query_generation(self, sample_crawl_request):
        """Test search query generation."""
        query = sample_crawl_request.get_search_query()
        assert "grade 8" in query
        assert "physics" in query
        assert "Work, Energy and Power" in query


class TestFormula:
    """Tests for Formula model."""
    
    def test_latex_validation(self):
        """Test LaTeX brace matching."""
        with pytest.raises(ValidationError):
            Formula(
                name="Invalid",
                latex="W = {F",  # Unmatched brace
                variables={}
            )
    
    def test_unicode_conversion(self, sample_formula):
        """Test LaTeX to Unicode conversion."""
        unicode = sample_formula.to_unicode()
        assert "×" in unicode  # \times → ×
        assert "θ" in unicode  # \theta → θ


class TestQuestionnaire:
    """Tests for Questionnaire model."""
    
    def test_question_count_validation(self):
        """Test that questionnaire must have exactly 50 questions."""
        # Create sections with wrong counts
        tier_1 = QuestionnaireSection(
            tier=QuestionTier.TIER_1,
            title="Tier 1",
            instructions="",
            questions=[]  # 0 questions, should be 20
        )
        tier_2 = QuestionnaireSection(
            tier=QuestionTier.TIER_2,
            title="Tier 2",
            instructions="",
            questions=[]
        )
        tier_3 = QuestionnaireSection(
            tier=QuestionTier.TIER_3,
            title="Tier 3",
            instructions="",
            questions=[]
        )
        
        with pytest.raises(ValidationError) as exc:
            Questionnaire(
                title="Test",
                subject=Subject.PHYSICS,
                grade=8,
                topic="Test",
                tier_1=tier_1,
                tier_2=tier_2,
                tier_3=tier_3
            )
        
        assert "must be 50" in str(exc.value).lower()
```

### 10.4 Integration Test Examples

```python
# tests/integration/test_crawl_pipeline.py

"""
Integration tests for the crawl pipeline.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from handlers.base import AbstractSourceHandler
from models import CrawlRequest, ExtractedContent, Subject


class TestCrawlPipeline:
    """Tests for the complete crawl pipeline."""
    
    @pytest.mark.asyncio
    async def test_single_source_extraction(
        self, 
        sample_crawl_request, 
        mock_khan_html,
        sample_extracted_content
    ):
        """Test extraction from a single source."""
        # Mock the handler
        with patch('handlers.khan.KhanAcademyHandler') as MockHandler:
            handler = MockHandler.return_value
            handler.can_handle.return_value = True
            handler.process = AsyncMock(return_value=sample_extracted_content)
            
            # Execute
            result = await handler.process(
                "https://khanacademy.org/science/physics/work"
            )
            
            # Verify
            assert result.total_content_items > 0
            assert result.quality_score >= 0.6
            assert len(result.concepts) > 0
    
    @pytest.mark.asyncio
    async def test_multi_source_aggregation(self, sample_crawl_request):
        """Test aggregation from multiple sources."""
        from processors.aggregator import ContentAggregator
        
        # Create mock content from multiple sources
        khan_content = ExtractedContent(
            source="khan_academy",
            url="https://khanacademy.org/...",
            concepts=[
                Concept(
                    title="Work Done",
                    definition="Work is force times displacement..."
                )
            ]
        )
        
        byjus_content = ExtractedContent(
            source="byjus",
            url="https://byjus.com/...",
            concepts=[
                Concept(
                    title="Work in Physics",  # Similar concept, different title
                    definition="Work is defined as force times distance..."
                )
            ]
        )
        
        aggregator = ContentAggregator(
            similarity_threshold=0.85
        )
        
        aggregated = aggregator.aggregate([khan_content, byjus_content])
        
        # Should deduplicate similar concepts
        assert len(aggregated.concepts) <= 2
        assert aggregated.duplicates_removed >= 0
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, sample_crawl_request):
        """Test fallback to alternative source on failure."""
        from agents.dispatcher import Dispatcher
        
        # Mock handlers where primary fails
        with patch('handlers.khan.KhanAcademyHandler') as MockKhan, \
             patch('handlers.byjus.ByjusHandler') as MockByjus:
            
            # Khan fails
            khan = MockKhan.return_value
            khan.can_handle.return_value = True
            khan.process = AsyncMock(side_effect=Exception("Timeout"))
            
            # Byjus succeeds
            byjus = MockByjus.return_value
            byjus.can_handle.return_value = True
            byjus.process = AsyncMock(return_value=sample_extracted_content)
            
            dispatcher = Dispatcher(handlers=[khan, byjus])
            results = await dispatcher.execute(sample_crawl_request)
            
            # Should have content from fallback
            assert len(results) > 0
            assert any(r.source == "byjus" for r in results)
```

### 10.5 Output Validation Tests

```python
# tests/validation/test_output_validation.py

"""
Tests for validating output against the validation checklist.
"""

import pytest
from models import (
    StudyMaterial, Questionnaire, Handout,
    StudyMaterialPage, ConceptBox,
    QuestionnaireSection, QuestionTier,
    MCQQuestion, AssertionReasoningQuestion,
    Subject, PedagogicalModel
)


class TestStudyMaterialValidation:
    """Validate study material against checklist requirements."""
    
    def test_minimum_pages(self, study_material):
        """Study material must have 4-7 pages."""
        assert 4 <= len(study_material.pages) <= 7
    
    def test_concept_boxes_per_page(self, study_material):
        """Each page must have 3-4 concept boxes."""
        for page in study_material.pages:
            assert 3 <= len(page.concept_boxes) <= 5, \
                f"Page {page.page_number} has {len(page.concept_boxes)} boxes"
    
    def test_concept_box_variety(self, study_material):
        """Each page must have concept_helper and misunderstanding boxes."""
        for page in study_material.pages:
            box_types = [box.box_type for box in page.concept_boxes]
            assert "concept_helper" in box_types, \
                f"Page {page.page_number} missing concept_helper box"
            assert "misunderstanding" in box_types, \
                f"Page {page.page_number} missing misunderstanding box"
    
    def test_pedagogical_framework(self, study_material):
        """Verify correct pedagogical framework for subject."""
        if study_material.subject in [Subject.PHYSICS, Subject.CHEMISTRY]:
            assert study_material.pedagogical_model == PedagogicalModel.FIVE_E
        elif study_material.subject == Subject.MATHEMATICS:
            assert study_material.pedagogical_model == PedagogicalModel.LES


class TestQuestionnaireValidation:
    """Validate questionnaire against checklist requirements."""
    
    def test_exactly_50_questions(self, questionnaire):
        """Questionnaire must have exactly 50 questions."""
        assert questionnaire.total_questions == 50, \
            f"Expected 50 questions, got {questionnaire.total_questions}"
    
    def test_tier_distribution(self, questionnaire):
        """Verify tier distribution: 20-20-10."""
        assert questionnaire.tier_1.question_count == 20
        assert questionnaire.tier_2.question_count == 20
        assert questionnaire.tier_3.question_count == 10
    
    def test_tier_1_composition(self, questionnaire):
        """Tier 1: 10 MCQ, 5 FIB, 5 T/F."""
        tier_1_questions = questionnaire.tier_1.questions
        
        mcq_count = sum(1 for q in tier_1_questions 
                       if q.question_type == "mcq")
        fib_count = sum(1 for q in tier_1_questions 
                       if q.question_type == "fill_blank")
        tf_count = sum(1 for q in tier_1_questions 
                      if q.question_type == "true_false")
        
        assert mcq_count == 10, f"Expected 10 MCQ, got {mcq_count}"
        assert fib_count == 5, f"Expected 5 FIB, got {fib_count}"
        assert tf_count == 5, f"Expected 5 T/F, got {tf_count}"
    
    def test_minimum_assertion_reasoning(self, questionnaire):
        """Tier 3 must have at least 6 assertion-reasoning questions."""
        ar_count = questionnaire.assertion_reasoning_count
        assert ar_count >= 6, \
            f"Expected at least 6 A-R questions, got {ar_count}"
    
    def test_answer_key_completeness(self, questionnaire):
        """All questions must have answers and explanations."""
        for question in questionnaire.get_all_questions():
            assert question.correct_answer, \
                f"Q{question.question_number} missing answer"
            
            # Tier 2 and 3 need explanations
            if question.tier in [QuestionTier.TIER_2, QuestionTier.TIER_3]:
                assert question.answer_explanation, \
                    f"Q{question.question_number} missing explanation"


class TestHandoutValidation:
    """Validate handout against checklist requirements."""
    
    def test_formula_table_present(self, handout):
        """Zone 1 must have formula table."""
        assert len(handout.formula_table) >= 3, \
            "Formula table must have at least 3 entries"
    
    def test_visual_map_present(self, handout):
        """Zone 2 must have visual map."""
        assert handout.visual_map_type is not None
        assert len(handout.visual_map_nodes) >= 3
    
    def test_pro_tips_present(self, handout):
        """Zone 3 must have pro tips."""
        assert len(handout.memory_tricks) >= 2
        assert len(handout.pro_formulas) >= 2
        assert len(handout.dont_confuse) >= 2
    
    def test_color_limit(self, handout):
        """Handout must use ≤5 colors."""
        assert len(handout.color_palette) <= 5
```

### 10.6 Test Configuration

```python
# tests/conftest.py

"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
from typing import Generator


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Environment configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        "rate_limit_interval": 0.1,  # Fast for testing
        "max_retries": 1,
        "timeout": 5,
        "mock_external_services": True,
    }


# Skip markers for different test types
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: marks tests that require external services"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that are slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# Auto-skip live tests in CI
def pytest_collection_modifyitems(config, items):
    if config.getoption("--ci"):
        skip_live = pytest.mark.skip(reason="Skipping live tests in CI")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
```

### 10.7 Running Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v --integration

# Run validation tests
pytest tests/validation -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run live tests (use sparingly)
pytest tests/live -v --live

# Run specific test file
pytest tests/unit/test_models.py -v

# Run tests matching pattern
pytest -k "questionnaire" -v

# Generate test report
pytest --html=report.html --self-contained-html
```

---

## 11. Deployment Checklist

### 11.1 Pre-Deployment

```markdown
- [ ] All unit tests pass
- [ ] All integration tests pass (mocked)
- [ ] Output validation tests pass
- [ ] NotebookLM authentication state saved
- [ ] Configuration file reviewed
- [ ] Rate limits configured appropriately
- [ ] Proxy configured for YouTube (if cloud deployment)
- [ ] Logging configured for production
- [ ] Error monitoring set up
```

### 11.2 Antigravity-Specific

```markdown
- [ ] Allowed domains configured in security policy
- [ ] JS execution policy set to "allow_list"
- [ ] Browser viewport meets minimum requirements (1280x800)
- [ ] Storage state file accessible
- [ ] Timeout configurations appropriate
```

### 11.3 Post-Deployment Verification

```markdown
- [ ] Test crawl request with each source
- [ ] Verify NotebookLM authentication works
- [ ] Check rate limiting is active
- [ ] Verify output generation for each type
- [ ] Monitor error rates for first hour
```

---

## 12. File Structure Summary

```
educrawler_specs/
├── SPECIFICATION.md              # Main specification (Part 1)
├── SPECIFICATION_PART2.md        # Error handling & testing (Part 2)
├── models/
│   ├── __init__.py              # Model exports
│   ├── request.py               # CrawlRequest, CrawlConfig
│   ├── content.py               # ExtractedContent, Concept, Formula
│   ├── output.py                # StudyMaterial, Questionnaire, Handout
│   └── state.py                 # CrawlState, ExecutionPlan
├── handlers/
│   ├── base.py                  # AbstractSourceHandler
│   ├── source_specs.py          # CSS selectors & configurations
│   ├── youtube.py               # YouTube transcript handler
│   └── notebooklm_integration.py # NotebookLM automation
├── templates/
│   ├── study_material.html      # Jinja2 template
│   ├── questionnaire.html       # Jinja2 template
│   └── handout.html             # Jinja2 template
├── config/
│   └── config.yaml              # Configuration schema
└── tests/
    ├── conftest.py              # Pytest configuration
    ├── fixtures/                # Test fixtures
    ├── unit/                    # Unit tests
    ├── integration/             # Integration tests
    └── validation/              # Output validation tests
```

---

**End of Phase 1 Specifications**

*Total files generated: 15+*
*Ready for Phase 2: Implementation in Claude Code*
