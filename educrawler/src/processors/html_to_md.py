"""
HTML to Markdown conversion processor.

Usage:
    from processors.html_to_md import html_to_markdown

    markdown = html_to_markdown(html_content)
"""

from __future__ import annotations

from markdownify import markdownify as md


def html_to_markdown(html: str, strip_tags: bool = True) -> str:
    """
    Convert HTML content to Markdown format.

    Args:
        html: HTML content to convert
        strip_tags: Whether to strip unnecessary tags

    Returns:
        Markdown formatted text
    """
    return md(html, strip=['script', 'style'] if strip_tags else None)
