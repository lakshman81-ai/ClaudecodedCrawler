"""
HTML to Markdown processor with LaTeX preservation and formula extraction.

Usage:
    from processors.html_to_md import html_to_markdown, extract_formulas

    markdown, formulas = html_to_markdown(html_content)
"""

from __future__ import annotations

import re
from typing import Tuple
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.models.content import Formula


def extract_formulas(html: str) -> list[Formula]:
    """
    Extract formulas from HTML containing LaTeX expressions.

    Searches for:
    - <script type="math/tex">...</script>
    - <annotation encoding="application/x-tex">...</annotation>

    Args:
        html: HTML content containing LaTeX formulas

    Returns:
        List of Formula objects
    """
    soup = BeautifulSoup(html, 'html.parser')
    formulas = []
    seen_latex = set()

    # Extract from <script type="math/tex"> tags
    for script in soup.find_all('script', type='math/tex'):
        latex = script.string
        if latex and latex.strip() and latex not in seen_latex:
            seen_latex.add(latex)
            formulas.append(Formula(
                name=_generate_formula_name(latex),
                latex=latex.strip()
            ))

    # Extract from <annotation encoding="application/x-tex"> tags
    for annotation in soup.find_all('annotation', encoding='application/x-tex'):
        latex = annotation.string
        if latex and latex.strip() and latex not in seen_latex:
            seen_latex.add(latex)
            formulas.append(Formula(
                name=_generate_formula_name(latex),
                latex=latex.strip()
            ))

    # Also check for inline LaTeX patterns like $...$ or $$...$$
    # Extract from text content
    text_content = soup.get_text()

    # Match display math $$...$$
    display_math = re.findall(r'\$\$(.+?)\$\$', text_content, re.DOTALL)
    for latex in display_math:
        latex = latex.strip()
        if latex and latex not in seen_latex:
            seen_latex.add(latex)
            formulas.append(Formula(
                name=_generate_formula_name(latex),
                latex=latex
            ))

    # Match inline math $...$
    inline_math = re.findall(r'(?<!\$)\$([^\$]+?)\$(?!\$)', text_content)
    for latex in inline_math:
        latex = latex.strip()
        if latex and latex not in seen_latex:
            seen_latex.add(latex)
            formulas.append(Formula(
                name=_generate_formula_name(latex),
                latex=latex
            ))

    return formulas


def _generate_formula_name(latex: str) -> str:
    """
    Generate a readable name from LaTeX expression.

    Args:
        latex: LaTeX expression

    Returns:
        Generated name
    """
    # Remove excessive whitespace
    latex = ' '.join(latex.split())

    # Truncate if too long
    if len(latex) > 100:
        return f"Formula: {latex[:97]}..."

    return f"Formula: {latex}"


def html_to_markdown(html: str, preserve_latex: bool = True) -> Tuple[str, list[Formula]]:
    """
    Convert HTML to Markdown while preserving LaTeX expressions and extracting formulas.

    Args:
        html: HTML content to convert
        preserve_latex: If True, preserve LaTeX expressions in the markdown

    Returns:
        Tuple of (markdown_text, list of Formula objects)
    """
    # First extract formulas before conversion
    formulas = extract_formulas(html)

    # If preserving LaTeX, protect math expressions before conversion
    protected_html = html
    math_placeholders = {}

    if preserve_latex:
        soup = BeautifulSoup(html, 'html.parser')

        # Protect <script type="math/tex"> tags
        for i, script in enumerate(soup.find_all('script', type='math/tex')):
            placeholder = f"MATHPLACEHOLDER{i}ENDMATH"
            latex = script.string or ""
            math_placeholders[placeholder] = f"${latex}$"
            script.replace_with(placeholder)

        # Protect <annotation encoding="application/x-tex"> tags
        for i, annotation in enumerate(soup.find_all('annotation', encoding='application/x-tex')):
            placeholder = f"ANNOTATIONPLACEHOLDER{i}ENDANNOTATION"
            latex = annotation.string or ""
            math_placeholders[placeholder] = f"${latex}$"
            annotation.replace_with(placeholder)

        protected_html = str(soup)

    # Convert HTML to Markdown
    markdown = md(
        protected_html,
        heading_style="ATX",
        bullets="-",
        code_language="",
        strip=['script[type!="math/tex"]', 'style', 'noscript']
    )

    # Restore protected LaTeX expressions
    for placeholder, latex_expr in math_placeholders.items():
        markdown = markdown.replace(placeholder, latex_expr)

    # Clean up excessive newlines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    # Clean up excessive whitespace
    markdown = re.sub(r'[ \t]+', ' ', markdown)

    return markdown.strip(), formulas


def clean_markdown(markdown: str) -> str:
    """
    Clean up markdown by removing excessive whitespace and fixing formatting.

    Args:
        markdown: Markdown text to clean

    Returns:
        Cleaned markdown
    """
    # Remove trailing whitespace on lines first
    markdown = re.sub(r'[ \t]+$', '', markdown, flags=re.MULTILINE)

    # Remove excessive newlines (3 or more becomes 2)
    while '\n\n\n' in markdown:
        markdown = markdown.replace('\n\n\n', '\n\n')

    # Ensure proper spacing around headers
    markdown = re.sub(r'(#{1,6} .+?)\n([^\n#])', r'\1\n\n\2', markdown)

    # Ensure lists have proper spacing
    markdown = re.sub(r'([^\n])\n([-*+] )', r'\1\n\n\2', markdown)

    return markdown.strip()
