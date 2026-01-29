"""
Demo script showing the processors in action.

Run this to see HTML to Markdown conversion and concept deduplication.
"""

from src.processors.html_to_md import html_to_markdown, extract_formulas
from src.processors.deduplicator import deduplicate_concepts
from src.models.content import Concept


def demo_html_to_markdown():
    """Demonstrate HTML to Markdown conversion with formula extraction."""
    print("=" * 60)
    print("HTML to Markdown Processor Demo")
    print("=" * 60)

    html = """
    <div class="lesson">
        <h1>Newton's Laws of Motion</h1>
        <p>Newton's second law states that force equals mass times acceleration:</p>
        <script type="math/tex">F = ma</script>

        <h2>Kinetic Energy</h2>
        <p>The kinetic energy formula is:</p>
        <script type="math/tex">KE = \\frac{1}{2}mv^2</script>

        <ul>
            <li>m = mass (kg)</li>
            <li>v = velocity (m/s)</li>
        </ul>
    </div>
    """

    markdown, formulas = html_to_markdown(html)

    print("\n📄 Markdown Output:")
    print("-" * 60)
    print(markdown)

    print("\n\n🔢 Extracted Formulas:")
    print("-" * 60)
    for i, formula in enumerate(formulas, 1):
        print(f"{i}. {formula.name}")
        print(f"   LaTeX: {formula.latex}")
        print()


def demo_deduplicator():
    """Demonstrate concept deduplication."""
    print("=" * 60)
    print("Concept Deduplicator Demo")
    print("=" * 60)

    concepts = [
        Concept(
            title="Work",
            definition="Work is the product of force and displacement in the direction of force",
            explanation="When a force moves an object through a distance, work is done on that object"
        ),
        Concept(
            title="Work",
            definition="Work is the product of force and displacement in the direction of force",
            explanation="When a force moves an object through a distance, work is done on the object"
        ),
        Concept(
            title="Energy",
            definition="Energy is the capacity to do work",
            explanation="Energy comes in many forms including kinetic and potential"
        ),
        Concept(
            title="Energy",
            definition="Energy is the ability to do work",
            explanation="Energy can take various forms such as kinetic and potential"
        ),
        Concept(
            title="Power",
            definition="Power is the rate of doing work",
            explanation="Power measures how quickly work is done"
        ),
    ]

    print(f"\n📚 Input: {len(concepts)} concepts")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept.title}: {concept.definition[:50]}...")

    # Deduplicate (threshold 0.70 = 70% similar)
    unique = deduplicate_concepts(concepts, threshold=0.70)

    print(f"\n✨ Output: {len(unique)} unique concepts (removed {len(concepts) - len(unique)} duplicates)")
    for i, concept in enumerate(unique, 1):
        print(f"{i}. {concept.title}: {concept.definition[:50]}...")


if __name__ == "__main__":
    demo_html_to_markdown()
    print("\n" * 2)
    demo_deduplicator()
    print("\n✅ Demo complete!")
