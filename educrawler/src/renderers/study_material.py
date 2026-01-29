"""
Study material renderer using Jinja2 templates.

Usage:
    from renderers.study_material import render_study_material

    html = render_study_material(study_material_obj)
"""

from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def get_template_env() -> Environment:
    """Get Jinja2 environment with template directory."""
    template_dir = Path(__file__).parent.parent.parent / "templates"
    return Environment(loader=FileSystemLoader(template_dir))


def render_study_material(study_material) -> str:
    """
    Render StudyMaterial to HTML.

    Args:
        study_material: StudyMaterial object

    Returns:
        Rendered HTML string
    """
    env = get_template_env()
    template = env.get_template("study_material.html")
    return template.render(material=study_material)
