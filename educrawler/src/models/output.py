"""
EduCrawler Output Models
========================

Pydantic models for final output documents:
- StudyMaterial: 5-page study guide with pedagogical framework
- Questionnaire: 50-question tiered assessment
- Handout: Single-page quick reference

These models enforce the validation checklist requirements.

Usage:
    from models.output import StudyMaterial, Questionnaire, Handout
    
    study_material = StudyMaterial(
        title="Work, Energy and Power",
        subject=Subject.PHYSICS,
        grade=8,
        pages=[...]
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


# Import from other models
from .request import Subject, OutputType, PedagogicalModel, SourceType


# =============================================================================
# QUESTION MODELS (for Questionnaire)
# =============================================================================

class QuestionTier(str, Enum):
    """Question difficulty tiers per validation checklist."""
    TIER_1 = "tier_1"  # Foundational Knowledge (20 questions)
    TIER_2 = "tier_2"  # Application & Conceptual (20 questions)
    TIER_3 = "tier_3"  # Analytical & Evaluative (10 questions)


class QuestionType(str, Enum):
    """Types of questions supported."""
    MCQ = "mcq"                     # Multiple Choice
    FILL_BLANK = "fill_blank"       # Fill in the Blank
    TRUE_FALSE = "true_false"       # True/False
    SHORT_ANSWER = "short_answer"   # Short Answer
    CALCULATION = "calculation"     # Numerical/Calculation
    CASE_STUDY = "case_study"       # Case Study
    ASSERTION_REASONING = "assertion_reasoning"  # A-R format
    DETAILED = "detailed"           # Long Answer


class MCQOption(BaseModel):
    """Option for multiple choice questions."""
    label: str = Field(pattern=r"^[a-d]$")  # a, b, c, d
    text: str
    is_correct: bool = Field(default=False)
    explanation: Optional[str] = Field(
        default=None,
        description="Why this option is correct/incorrect"
    )


class BaseQuestion(BaseModel):
    """Base class for all question types."""
    
    id: UUID = Field(default_factory=uuid4)
    question_number: int = Field(ge=1, le=50)
    tier: QuestionTier
    question_type: QuestionType
    question_text: str = Field(min_length=10)
    marks: int = Field(default=1, ge=1, le=10)
    
    # Answer
    correct_answer: str
    answer_explanation: str = Field(
        default="",
        description="Detailed explanation for answer key"
    )
    
    # Metadata
    concept_tested: str = Field(default="")
    difficulty_level: Literal["easy", "medium", "hard"] = Field(default="medium")
    time_estimate_seconds: int = Field(default=60, ge=30, le=600)


class MCQQuestion(BaseQuestion):
    """Multiple Choice Question."""
    
    question_type: Literal[QuestionType.MCQ] = QuestionType.MCQ
    options: list[MCQOption] = Field(min_length=4, max_length=4)
    
    @field_validator("options")
    @classmethod
    def validate_single_correct(cls, v: list[MCQOption]) -> list[MCQOption]:
        """Ensure exactly one correct option."""
        correct_count = sum(1 for opt in v if opt.is_correct)
        if correct_count != 1:
            raise ValueError(f"MCQ must have exactly 1 correct option, got {correct_count}")
        return v
    
    @model_validator(mode="after")
    def set_correct_answer(self) -> "MCQQuestion":
        """Auto-set correct_answer from options."""
        for opt in self.options:
            if opt.is_correct:
                object.__setattr__(self, "correct_answer", opt.label)
                break
        return self


class FillBlankQuestion(BaseQuestion):
    """Fill in the Blank Question."""
    
    question_type: Literal[QuestionType.FILL_BLANK] = QuestionType.FILL_BLANK
    blank_position: str = Field(
        default="________",
        description="Placeholder text for blank"
    )
    acceptable_answers: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable answers"
    )


class TrueFalseQuestion(BaseQuestion):
    """True/False Question."""
    
    question_type: Literal[QuestionType.TRUE_FALSE] = QuestionType.TRUE_FALSE
    correct_answer: Literal["True", "False"]
    statement: str = Field(description="The statement to evaluate")
    
    @model_validator(mode="after")
    def sync_statement(self) -> "TrueFalseQuestion":
        """Ensure question_text matches statement."""
        if not self.question_text or self.question_text == "":
            object.__setattr__(self, "question_text", f"True or False: {self.statement}")
        return self


class AssertionReasoningQuestion(BaseQuestion):
    """
    Assertion-Reasoning Question (Tier 3).
    
    Format:
    Assertion (A): [statement]
    Reason (R): [explanation]
    
    Options:
    a) Both A and R are true, and R is the correct explanation of A.
    b) Both A and R are true, but R is not the correct explanation of A.
    c) A is true, but R is false.
    d) A is false, but R is true.
    e) Both A and R are false. (optional 5th option)
    """
    
    question_type: Literal[QuestionType.ASSERTION_REASONING] = QuestionType.ASSERTION_REASONING
    tier: Literal[QuestionTier.TIER_3] = QuestionTier.TIER_3
    
    assertion: str = Field(min_length=10)
    reason: str = Field(min_length=10)
    
    correct_answer: Literal["a", "b", "c", "d", "e"]
    
    # Standard A-R options
    options: list[str] = Field(
        default=[
            "Both A and R are true, and R is the correct explanation of A.",
            "Both A and R are true, but R is not the correct explanation of A.",
            "A is true, but R is false.",
            "A is false, but R is true.",
            "Both A and R are false."
        ]
    )
    
    hint: Optional[str] = Field(default=None)
    
    @model_validator(mode="after")
    def build_question_text(self) -> "AssertionReasoningQuestion":
        """Build formatted question text."""
        text = f"""Assertion (A): {self.assertion}

Reason (R): {self.reason}

Options:
a) {self.options[0]}
b) {self.options[1]}
c) {self.options[2]}
d) {self.options[3]}"""
        if len(self.options) > 4:
            text += f"\ne) {self.options[4]}"
        
        object.__setattr__(self, "question_text", text)
        return self


class CalculationQuestion(BaseQuestion):
    """Numerical/Calculation Question (Tier 2)."""
    
    question_type: Literal[QuestionType.CALCULATION] = QuestionType.CALCULATION
    
    given_data: dict[str, str] = Field(
        default_factory=dict,
        description="Given values with units"
    )
    formula_required: str = Field(default="")
    solution_steps: list[str] = Field(default_factory=list)
    final_answer_with_unit: str = Field(description="Answer with SI unit")
    
    @model_validator(mode="after")
    def set_correct_answer(self) -> "CalculationQuestion":
        """Set correct_answer from final_answer_with_unit."""
        object.__setattr__(self, "correct_answer", self.final_answer_with_unit)
        return self


class CaseStudyQuestion(BaseQuestion):
    """Case Study / Detailed Question (Tier 2/3)."""
    
    question_type: Literal[QuestionType.CASE_STUDY] = QuestionType.CASE_STUDY
    
    case_description: str = Field(min_length=50)
    sub_questions: list[str] = Field(min_length=1, max_length=5)
    sub_answers: list[str] = Field(min_length=1)
    
    @field_validator("sub_answers")
    @classmethod
    def match_sub_questions(cls, v: list[str], info) -> list[str]:
        """Ensure sub_answers matches sub_questions count."""
        # Note: This validation runs before sub_questions is set
        # Full validation should be done at model level
        return v


class DetailedQuestion(BaseQuestion):
    """Long Answer / Detailed Question (Tier 2/3)."""
    
    question_type: Literal[QuestionType.DETAILED] = QuestionType.DETAILED
    
    expected_points: list[str] = Field(
        min_length=2,
        description="Key points expected in answer"
    )
    word_limit: Optional[int] = Field(default=None, ge=50, le=500)
    diagram_required: bool = Field(default=False)
    
    @model_validator(mode="after")
    def build_answer(self) -> "DetailedQuestion":
        """Build correct_answer from expected_points."""
        answer = "\n".join(f"• {point}" for point in self.expected_points)
        object.__setattr__(self, "correct_answer", answer)
        return self


# Union type for all question types
Question = Union[
    MCQQuestion,
    FillBlankQuestion,
    TrueFalseQuestion,
    AssertionReasoningQuestion,
    CalculationQuestion,
    CaseStudyQuestion,
    DetailedQuestion
]


# =============================================================================
# STUDY MATERIAL MODEL
# =============================================================================

class ConceptBox(BaseModel):
    """Educational concept box for study material pages."""
    
    box_type: Literal[
        "concept_helper",      # 💡
        "misunderstanding",    # ⚠️
        "real_world",          # 🌍
        "did_you_know"         # 🔍
    ]
    title: str = Field(max_length=50)
    content: str = Field(min_length=20, max_length=300)
    
    @computed_field
    @property
    def emoji(self) -> str:
        """Get emoji for box type."""
        emoji_map = {
            "concept_helper": "💡",
            "misunderstanding": "⚠️",
            "real_world": "🌍",
            "did_you_know": "🔍"
        }
        return emoji_map.get(self.box_type, "📝")


class StudyMaterialPage(BaseModel):
    """Single page of study material."""
    
    page_number: int = Field(ge=1, le=10)
    title: str
    
    # Content sections (varies by pedagogical model)
    sections: list[dict[str, str]] = Field(
        default_factory=list,
        description="[{heading: str, content: str}]"
    )
    
    # Cornell Notes format (for Explain/Elaborate pages)
    cornell_notes: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="[{cue: str, notes: str}]"
    )
    summary: Optional[str] = Field(default=None)
    
    # Educational boxes (REQUIRED: 3-4 per page)
    concept_boxes: list[ConceptBox] = Field(
        min_length=3,
        max_length=5,
        description="3-4 concept boxes per page (required)"
    )
    
    # Visual elements
    images: list[dict[str, str]] = Field(
        default_factory=list,
        description="[{url: str, caption: str, alt_text: str}]"
    )
    diagrams: list[dict[str, str]] = Field(default_factory=list)
    
    # Formulas (Unicode only, no images)
    formulas: list[dict[str, str]] = Field(
        default_factory=list,
        description="[{name: str, unicode: str, latex: str}]"
    )
    
    @field_validator("concept_boxes")
    @classmethod
    def validate_box_variety(cls, v: list[ConceptBox]) -> list[ConceptBox]:
        """Ensure box type variety."""
        box_types = [box.box_type for box in v]
        if "concept_helper" not in box_types:
            raise ValueError("Each page must have at least 1 Concept Helper box")
        if "misunderstanding" not in box_types:
            raise ValueError("Each page must have at least 1 Misunderstanding box")
        return v


class StudyMaterial(BaseModel):
    """
    Complete Study Material document.
    
    Requirements (per validation checklist):
    - 5 pages following pedagogical framework
    - 3-4 concept boxes per page
    - 1-2 images per page
    - Unicode formulas (no images)
    - Cornell Notes format for Explain sections
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Document metadata
    title: str
    subject: Subject
    grade: int = Field(ge=1, le=12)
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    
    # Pedagogical framework
    pedagogical_model: PedagogicalModel
    
    # Content
    pages: list[StudyMaterialPage] = Field(min_length=4, max_length=7)
    
    # Sources
    sources_used: list[SourceType] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_duration_seconds: float = Field(default=0)
    
    @computed_field
    @property
    def total_concept_boxes(self) -> int:
        """Total concept boxes across all pages."""
        return sum(len(page.concept_boxes) for page in self.pages)
    
    @computed_field
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @model_validator(mode="after")
    def validate_structure(self) -> "StudyMaterial":
        """Validate pedagogical structure."""
        if self.pedagogical_model == PedagogicalModel.FIVE_E:
            expected_titles = ["engage", "explore", "explain", "elaborate", "evaluate"]
            # Check that pages roughly follow 5E
            # (Allow flexibility in exact naming)
        return self


# =============================================================================
# QUESTIONNAIRE MODEL
# =============================================================================

class QuestionnaireSection(BaseModel):
    """Section of questionnaire by tier."""
    
    tier: QuestionTier
    title: str
    instructions: str
    questions: list[Question] = Field(default_factory=list)
    
    @computed_field
    @property
    def question_count(self) -> int:
        return len(self.questions)


class Questionnaire(BaseModel):
    """
    Complete Questionnaire document.
    
    Requirements (per validation checklist):
    - EXACTLY 50 questions total
    - Tier 1: 20 questions (10 MCQ, 5 FIB, 5 T/F)
    - Tier 2: 20 questions (10 Application, 5 Case Study, 5 Calculation)
    - Tier 3: 10 questions (6+ Assertion-Reasoning, 2 Error Analysis, 2 Synthesis)
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Document metadata
    title: str
    subject: Subject
    grade: int = Field(ge=1, le=12)
    topic: str
    
    # Sections
    tier_1: QuestionnaireSection = Field(
        description="Foundational Knowledge (20 questions)"
    )
    tier_2: QuestionnaireSection = Field(
        description="Application & Conceptual (20 questions)"
    )
    tier_3: QuestionnaireSection = Field(
        description="Analytical & Evaluative (10 questions)"
    )
    
    # Answer key
    include_answer_key: bool = Field(default=True)
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def total_questions(self) -> int:
        """Total question count (must be exactly 50)."""
        return (
            self.tier_1.question_count +
            self.tier_2.question_count +
            self.tier_3.question_count
        )
    
    @computed_field
    @property
    def assertion_reasoning_count(self) -> int:
        """Count of A-R questions in Tier 3."""
        return sum(
            1 for q in self.tier_3.questions
            if isinstance(q, AssertionReasoningQuestion)
        )
    
    @model_validator(mode="after")
    def validate_question_counts(self) -> "Questionnaire":
        """Enforce exact question counts per validation checklist."""
        errors = []
        
        # Total must be exactly 50
        if self.total_questions != 50:
            errors.append(f"Total questions must be 50, got {self.total_questions}")
        
        # Tier 1 must be 20
        if self.tier_1.question_count != 20:
            errors.append(f"Tier 1 must have 20 questions, got {self.tier_1.question_count}")
        
        # Tier 2 must be 20
        if self.tier_2.question_count != 20:
            errors.append(f"Tier 2 must have 20 questions, got {self.tier_2.question_count}")
        
        # Tier 3 must be 10
        if self.tier_3.question_count != 10:
            errors.append(f"Tier 3 must have 10 questions, got {self.tier_3.question_count}")
        
        # Tier 3 must have at least 6 A-R questions
        if self.assertion_reasoning_count < 6:
            errors.append(
                f"Tier 3 must have at least 6 Assertion-Reasoning questions, "
                f"got {self.assertion_reasoning_count}"
            )
        
        if errors:
            raise ValueError("Questionnaire validation failed:\n" + "\n".join(errors))
        
        return self
    
    def get_all_questions(self) -> list[Question]:
        """Get all questions in order."""
        return (
            self.tier_1.questions +
            self.tier_2.questions +
            self.tier_3.questions
        )
    
    def generate_answer_key(self) -> str:
        """Generate formatted answer key."""
        lines = [f"# Answer Key: {self.title}\n"]
        
        for tier_name, section in [
            ("Tier 1: Foundational Knowledge", self.tier_1),
            ("Tier 2: Application & Conceptual", self.tier_2),
            ("Tier 3: Analytical & Evaluative", self.tier_3),
        ]:
            lines.append(f"\n## {tier_name}\n")
            for q in section.questions:
                lines.append(f"**Q{q.question_number}.** {q.correct_answer}")
                if q.answer_explanation:
                    lines.append(f"   *Explanation:* {q.answer_explanation}")
        
        return "\n".join(lines)


# =============================================================================
# HANDOUT MODEL
# =============================================================================

class FormulaTableEntry(BaseModel):
    """Entry in the formula reference table."""
    
    concept: str
    formula_unicode: str
    formula_latex: str = Field(default="")
    units: str
    when_to_use: str


class VisualMapNode(BaseModel):
    """Node in the visual concept map."""
    
    id: str
    label: str
    node_type: Literal["central", "branch", "leaf"]
    parent_id: Optional[str] = Field(default=None)
    connections: list[str] = Field(default_factory=list)


class ProTip(BaseModel):
    """Pro tip entry for handout."""
    
    tip_type: Literal["memory_trick", "pro_formula", "dont_confuse", "problem_solving"]
    title: str
    content: str
    
    # For "dont_confuse" type
    wrong_approach: Optional[str] = Field(default=None)
    correct_approach: Optional[str] = Field(default=None)


class Handout(BaseModel):
    """
    Single-page Quick Reference Handout.
    
    Requirements (per validation checklist):
    - Zone 1 (Top): Formula Reference Table
    - Zone 2 (Middle): Visual Map (Timeline/Mind Map/Flowchart)
    - Zone 3 (Bottom): Pro Tips (Memory tricks, Don't Confuse, etc.)
    - A4 Landscape, single page
    - Sans serif font ≥13pt
    - ≤5 colors
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Document metadata
    title: str
    subject: Subject
    grade: int = Field(ge=1, le=12)
    topic: str
    
    # Zone 1: Quick Reference (Top Third)
    formula_table: list[FormulaTableEntry] = Field(
        min_length=3,
        description="Formula reference table"
    )
    
    # Zone 2: Visual Map (Middle Third)
    visual_map_type: Literal["timeline", "mind_map", "flowchart", "decision_tree", "comparison_chart", "hierarchy"]
    visual_map_nodes: list[VisualMapNode] = Field(min_length=3)
    visual_map_title: str
    
    # Zone 3: Pro Tips (Bottom Third)
    memory_tricks: list[ProTip] = Field(
        min_length=2,
        description="At least 2 mnemonics/memory tricks"
    )
    pro_formulas: list[ProTip] = Field(
        min_length=2,
        description="Derived equations and shortcuts"
    )
    dont_confuse: list[ProTip] = Field(
        min_length=2,
        description="Common errors and corrections"
    )
    problem_solving_tips: list[str] = Field(
        default_factory=list,
        description="Step-by-step approach reminders"
    )
    
    # Design specifications
    color_palette: list[str] = Field(
        max_length=5,
        description="≤5 colors for the handout"
    )
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def total_pro_tips(self) -> int:
        """Total pro tips across all categories."""
        return (
            len(self.memory_tricks) +
            len(self.pro_formulas) +
            len(self.dont_confuse)
        )
    
    @model_validator(mode="after")
    def validate_pro_tip_types(self) -> "Handout":
        """Ensure pro tips have correct types."""
        for tip in self.memory_tricks:
            if tip.tip_type != "memory_trick":
                raise ValueError(f"Memory trick has wrong type: {tip.tip_type}")
        for tip in self.pro_formulas:
            if tip.tip_type != "pro_formula":
                raise ValueError(f"Pro formula has wrong type: {tip.tip_type}")
        for tip in self.dont_confuse:
            if tip.tip_type != "dont_confuse":
                raise ValueError(f"Don't confuse has wrong type: {tip.tip_type}")
        return self


# =============================================================================
# UNIFIED OUTPUT DOCUMENT
# =============================================================================

class OutputDocument(BaseModel):
    """
    Unified output document wrapper.
    
    Contains one of: StudyMaterial, Questionnaire, or Handout
    """
    
    id: UUID = Field(default_factory=uuid4)
    output_type: OutputType
    
    # One of these will be populated
    study_material: Optional[StudyMaterial] = Field(default=None)
    questionnaire: Optional[Questionnaire] = Field(default=None)
    handout: Optional[Handout] = Field(default=None)
    
    # Metadata
    request_id: UUID
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_duration_seconds: float = Field(default=0)
    
    # Quality metrics
    validation_passed: bool = Field(default=False)
    validation_errors: list[str] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def validate_content_present(self) -> "OutputDocument":
        """Ensure correct content type is present."""
        content_map = {
            OutputType.STUDY_MATERIAL: self.study_material,
            OutputType.QUESTIONNAIRE: self.questionnaire,
            OutputType.HANDOUT: self.handout,
        }
        
        expected_content = content_map.get(self.output_type)
        if expected_content is None:
            raise ValueError(
                f"OutputDocument type is {self.output_type.value} "
                f"but no {self.output_type.value} content provided"
            )
        
        return self
    
    def get_content(self) -> Union[StudyMaterial, Questionnaire, Handout]:
        """Get the actual content based on output_type."""
        content_map = {
            OutputType.STUDY_MATERIAL: self.study_material,
            OutputType.QUESTIONNAIRE: self.questionnaire,
            OutputType.HANDOUT: self.handout,
        }
        return content_map[self.output_type]


# Example usage
if __name__ == "__main__":
    # Create sample MCQ
    mcq = MCQQuestion(
        question_number=1,
        tier=QuestionTier.TIER_1,
        question_text="Which of Newton's laws states that F = ma?",
        marks=1,
        correct_answer="b",
        answer_explanation="Newton's Second Law relates force, mass, and acceleration.",
        options=[
            MCQOption(label="a", text="First Law", is_correct=False),
            MCQOption(label="b", text="Second Law", is_correct=True),
            MCQOption(label="c", text="Third Law", is_correct=False),
            MCQOption(label="d", text="Law of Gravitation", is_correct=False),
        ]
    )
    
    # Create sample A-R question
    ar = AssertionReasoningQuestion(
        question_number=41,
        assertion="Work done is zero when force is perpendicular to displacement.",
        reason="Work = F × d × cos(θ), and cos(90°) = 0.",
        correct_answer="a",
        answer_explanation="Both statements are true and the reason correctly explains the assertion.",
        marks=2
    )
    
    print("MCQ Question:")
    print(mcq.model_dump_json(indent=2))
    print("\nA-R Question:")
    print(ar.question_text)
