"""Tests for content processors."""

import pytest
from src.models.content import Concept, Formula
from src.processors.html_to_md import html_to_markdown, extract_formulas, clean_markdown
from src.processors.deduplicator import ConceptDeduplicator, deduplicate_concepts


class TestHTMLToMarkdown:
    """Tests for HTML to Markdown processor."""

    def test_basic_html_conversion(self):
        """Test basic HTML to Markdown conversion."""
        html = "<h1>Physics</h1><p>Study of matter and energy.</p>"
        markdown, formulas = html_to_markdown(html)

        assert "# Physics" in markdown
        assert "Study of matter and energy" in markdown
        assert len(formulas) == 0

    def test_formula_extraction_from_script_tags(self):
        """Test extracting formulas from script type='math/tex' tags."""
        html = """
        <p>Einstein's equation:</p>
        <script type="math/tex">E = mc^2</script>
        """
        markdown, formulas = html_to_markdown(html)

        assert len(formulas) == 1
        assert formulas[0].latex == "E = mc^2"
        assert "Formula:" in formulas[0].name

    def test_formula_extraction_from_annotation_tags(self):
        """Test extracting formulas from annotation tags."""
        html = """
        <p>Newton's second law:</p>
        <annotation encoding="application/x-tex">F = ma</annotation>
        """
        markdown, formulas = html_to_markdown(html)

        assert len(formulas) == 1
        assert formulas[0].latex == "F = ma"

    def test_formula_extraction_from_dollar_signs(self):
        """Test extracting formulas from $...$ and $$...$$ patterns."""
        html = """
        <p>Inline math: $v = u + at$ and display math:</p>
        <p>$$s = ut + \\frac{1}{2}at^2$$</p>
        """
        markdown, formulas = html_to_markdown(html)

        assert len(formulas) >= 2
        latex_formulas = [f.latex for f in formulas]
        assert "v = u + at" in latex_formulas
        assert "s = ut + \\frac{1}{2}at^2" in latex_formulas

    def test_latex_preservation(self):
        """Test that LaTeX is preserved in markdown output."""
        html = """
        <p>The formula is:</p>
        <script type="math/tex">\\int_0^\\infty e^{-x} dx = 1</script>
        """
        markdown, formulas = html_to_markdown(html, preserve_latex=True)

        # Check that LaTeX appears in markdown (as $...$)
        assert "$" in markdown or "int" in markdown.lower()

    def test_no_duplicate_formulas(self):
        """Test that duplicate formulas are not extracted."""
        html = """
        <script type="math/tex">E = mc^2</script>
        <script type="math/tex">E = mc^2</script>
        <p>$$E = mc^2$$</p>
        """
        markdown, formulas = html_to_markdown(html)

        # Should only have one instance
        assert len(formulas) == 1
        assert formulas[0].latex == "E = mc^2"

    def test_complex_html_structure(self):
        """Test conversion of complex HTML with nested elements."""
        html = """
        <div class="content">
            <h2>Work and Energy</h2>
            <ul>
                <li>Work is force times distance</li>
                <li>Energy is the ability to do work</li>
            </ul>
            <p>Formula: <script type="math/tex">W = Fd\\cos\\theta</script></p>
        </div>
        """
        markdown, formulas = html_to_markdown(html)

        assert "## Work and Energy" in markdown
        assert "Work is force times distance" in markdown
        assert "Energy is the ability to do work" in markdown
        assert len(formulas) == 1
        assert "W = Fd" in formulas[0].latex

    def test_clean_markdown(self):
        """Test markdown cleaning function."""
        dirty = "# Header\n\n\n\nParagraph   \n  \n\n- Item"
        clean = clean_markdown(dirty)

        # Should reduce multiple newlines
        assert "\n\n\n" not in clean
        # Should remove trailing whitespace
        assert "   \n" not in clean


class TestConceptDeduplicator:
    """Tests for concept deduplicator."""

    def test_no_duplicates(self):
        """Test deduplication with completely unique concepts."""
        concepts = [
            Concept(
                title="Kinetic Energy",
                definition="Energy of motion",
                explanation="Energy possessed by a moving object"
            ),
            Concept(
                title="Potential Energy",
                definition="Stored energy",
                explanation="Energy stored due to position or configuration"
            ),
            Concept(
                title="Thermal Energy",
                definition="Heat energy",
                explanation="Energy from random molecular motion"
            )
        ]

        dedup = ConceptDeduplicator(threshold=0.85)
        result = dedup.deduplicate_concepts(concepts)

        assert len(result) == 3

    def test_exact_duplicates(self):
        """Test deduplication of exact duplicate concepts."""
        concept1 = Concept(
            title="Work",
            definition="Force applied over distance",
            explanation="Work is done when force moves an object"
        )
        concept2 = Concept(
            title="Work",
            definition="Force applied over distance",
            explanation="Work is done when force moves an object"
        )

        concepts = [concept1, concept2]
        dedup = ConceptDeduplicator(threshold=0.85)
        result = dedup.deduplicate_concepts(concepts)

        # Should keep only one
        assert len(result) == 1
        assert result[0].title == "Work"

    def test_similar_concepts(self):
        """Test deduplication of very similar but not identical concepts."""
        # Create concepts with high textual overlap
        shared_text = (
            "Momentum is a fundamental concept in physics that describes "
            "the motion of objects. It is defined as the product of mass and velocity. "
            "Momentum is a vector quantity conserved in closed systems."
        )
        concept1 = Concept(
            title="Momentum",
            definition=shared_text,
            explanation=shared_text + " This is an important conservation law.",
            key_points=["Vector quantity", "Conserved", "Mass times velocity"]
        )
        concept2 = Concept(
            title="Momentum",
            definition=shared_text,
            explanation=shared_text + " It's a key conservation principle.",
            key_points=["Vector quantity", "Conserved", "Mass times velocity"]
        )

        concepts = [concept1, concept2]

        # Check similarity
        dedup = ConceptDeduplicator(threshold=0.75)
        similarity = dedup.calculate_similarity(concept1, concept2)

        # With such high overlap, similarity should be high
        assert similarity > 0.7, f"Expected high similarity, got {similarity:.2f}"

        # Lower threshold should catch them
        dedup_low = ConceptDeduplicator(threshold=0.60)
        result = dedup_low.deduplicate_concepts(concepts)
        assert len(result) == 1, "Very similar concepts should be deduplicated"

    def test_different_concepts(self):
        """Test that different concepts are kept separate."""
        concept1 = Concept(
            title="Speed",
            definition="Rate of change of distance",
            explanation="Speed is how fast something moves"
        )
        concept2 = Concept(
            title="Velocity",
            definition="Rate of change of displacement",
            explanation="Velocity is speed with direction"
        )

        concepts = [concept1, concept2]
        dedup = ConceptDeduplicator(threshold=0.85)
        result = dedup.deduplicate_concepts(concepts)

        # Should keep both (not similar enough)
        assert len(result) == 2

    def test_threshold_adjustment(self):
        """Test that threshold affects deduplication."""
        concept1 = Concept(
            title="Energy",
            definition="Ability to do work",
            explanation="Energy can take many forms"
        )
        concept2 = Concept(
            title="Energy",
            definition="Capacity for doing work",
            explanation="Energy exists in various forms"
        )

        concepts = [concept1, concept2]

        # With high threshold (0.95), might keep both
        dedup_strict = ConceptDeduplicator(threshold=0.95)
        result_strict = dedup_strict.deduplicate_concepts(concepts)

        # With low threshold (0.70), should merge
        dedup_loose = ConceptDeduplicator(threshold=0.70)
        result_loose = dedup_loose.deduplicate_concepts(concepts)

        # Loose threshold should find more duplicates
        assert len(result_loose) <= len(result_strict)

    def test_empty_list(self):
        """Test deduplication with empty list."""
        dedup = ConceptDeduplicator()
        result = dedup.deduplicate_concepts([])
        assert result == []

    def test_single_concept(self):
        """Test deduplication with single concept."""
        concept = Concept(
            title="Force",
            definition="Push or pull"
        )
        dedup = ConceptDeduplicator()
        result = dedup.deduplicate_concepts([concept])
        assert len(result) == 1

    def test_similarity_calculation(self):
        """Test similarity calculation between concepts."""
        concept1 = Concept(
            title="Acceleration",
            definition="Rate of change of velocity",
            explanation="How quickly velocity changes"
        )
        concept2 = Concept(
            title="Acceleration",
            definition="Rate of change of velocity",
            explanation="How quickly velocity changes"
        )
        concept3 = Concept(
            title="Mass",
            definition="Amount of matter",
            explanation="Measure of inertia"
        )

        dedup = ConceptDeduplicator()

        # Identical concepts should have similarity ~1.0
        sim_identical = dedup.calculate_similarity(concept1, concept2)
        assert sim_identical > 0.9

        # Different concepts should have low similarity
        sim_different = dedup.calculate_similarity(concept1, concept3)
        assert sim_different < 0.5

    def test_convenience_function(self):
        """Test the convenience deduplicate_concepts function."""
        concepts = [
            Concept(title="A", definition="First concept"),
            Concept(title="A", definition="First concept"),
            Concept(title="B", definition="Second concept")
        ]

        result = deduplicate_concepts(concepts, threshold=0.85)
        assert len(result) == 2


def test_processors_import():
    """Test that all processors can be imported."""
    from src.processors.html_to_md import html_to_markdown
    from src.processors.deduplicator import ConceptDeduplicator

    assert callable(html_to_markdown)
    assert ConceptDeduplicator is not None
