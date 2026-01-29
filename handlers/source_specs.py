"""
EduCrawler Source Handler Specifications
========================================

Detailed specifications for each source handler including:
- Base URL patterns
- CSS/XPath selectors for content extraction
- Rate limiting requirements
- Error handling specifics
- Sample extraction mappings

This module serves as the configuration reference for implementing
source-specific handlers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelectorSpec:
    """CSS/XPath selector specification."""
    css: str
    xpath: Optional[str] = None
    attribute: Optional[str] = None  # For extracting attributes instead of text
    multiple: bool = False  # Whether to select multiple elements
    required: bool = True
    fallback_css: Optional[str] = None


@dataclass
class RateLimitSpec:
    """Rate limiting specification."""
    min_interval_seconds: float
    max_requests_per_minute: int
    burst_allowed: int = 1
    respect_retry_after: bool = True


@dataclass
class RetrySpec:
    """Retry specification."""
    max_retries: int
    backoff_base: float
    backoff_max_seconds: float
    retry_on_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class SourceHandlerSpec:
    """Complete specification for a source handler."""
    
    name: str
    domain: str
    base_urls: list[str]
    
    # Content selectors
    selectors: dict[str, SelectorSpec]
    
    # Rate limiting
    rate_limit: RateLimitSpec
    
    # Retry configuration
    retry: RetrySpec
    
    # Browser requirements
    requires_js_rendering: bool = True
    requires_mathjax_wait: bool = False
    cloudflare_protected: bool = False
    
    # Authentication
    requires_auth: bool = False
    
    # Special handling
    has_pdf_downloads: bool = False
    sitemap_url: Optional[str] = None
    robots_txt_notes: str = ""


# =============================================================================
# KHAN ACADEMY SPECIFICATION
# =============================================================================

KHAN_ACADEMY_SPEC = SourceHandlerSpec(
    name="Khan Academy",
    domain="khanacademy.org",
    base_urls=[
        "https://www.khanacademy.org/science/physics",
        "https://www.khanacademy.org/science/chemistry",
        "https://www.khanacademy.org/math",
        "https://www.khanacademy.org/science/biology",
    ],
    selectors={
        # Main content
        "article_content": SelectorSpec(
            css="[data-test-id='article-renderer'] .paragraph, .perseus-article",
            xpath="//div[contains(@class, 'article-renderer')]//p",
            multiple=True
        ),
        "lesson_title": SelectorSpec(
            css="h1[data-test-id='lesson-title'], .lesson-title",
            xpath="//h1[contains(@class, 'lesson-title')]"
        ),
        "video_container": SelectorSpec(
            css="[data-test-id='video-player'], .video-js",
            required=False
        ),
        "youtube_video_id": SelectorSpec(
            css="[data-youtube-id]",
            attribute="data-youtube-id",
            required=False
        ),
        
        # Math content
        "math_expressions": SelectorSpec(
            css=".MathJax, .katex, [data-math-formula]",
            multiple=True,
            required=False
        ),
        "math_latex": SelectorSpec(
            css="script[type='math/tex'], annotation[encoding='application/x-tex']",
            multiple=True,
            required=False
        ),
        
        # Exercise content
        "exercise_container": SelectorSpec(
            css=".perseus-renderer, [data-test-id='exercise-content']",
            required=False
        ),
        "exercise_question": SelectorSpec(
            css=".perseus-renderer .paragraph",
            multiple=True,
            required=False
        ),
        "exercise_choices": SelectorSpec(
            css=".perseus-radio-option, .choice-clue",
            multiple=True,
            required=False
        ),
        
        # Navigation
        "breadcrumb": SelectorSpec(
            css="nav[aria-label='breadcrumb'] a, .breadcrumb a",
            multiple=True,
            required=False
        ),
        "related_content": SelectorSpec(
            css="[data-test-id='related-content'] a",
            attribute="href",
            multiple=True,
            required=False
        ),
    },
    rate_limit=RateLimitSpec(
        min_interval_seconds=2.0,
        max_requests_per_minute=20,
        burst_allowed=2
    ),
    retry=RetrySpec(
        max_retries=3,
        backoff_base=2.0,
        backoff_max_seconds=16.0
    ),
    requires_js_rendering=True,
    requires_mathjax_wait=True,
    cloudflare_protected=False,
    sitemap_url="https://www.khanacademy.org/sitemap.xml",
    robots_txt_notes="GPTBot blocked, general crawling allowed for educational content"
)


# =============================================================================
# BYJUS SPECIFICATION
# =============================================================================

BYJUS_SPEC = SourceHandlerSpec(
    name="Byjus",
    domain="byjus.com",
    base_urls=[
        "https://byjus.com/physics/",
        "https://byjus.com/chemistry/",
        "https://byjus.com/maths/",
        "https://byjus.com/biology/",
        "https://byjus.com/cbse-notes/",
    ],
    selectors={
        # Main content
        "article_content": SelectorSpec(
            css=".article-content, .post-content, .entry-content",
            fallback_css="article p, .content-wrapper p",
            multiple=True
        ),
        "page_title": SelectorSpec(
            css="h1.entry-title, h1.article-title, .page-title h1",
            fallback_css="h1"
        ),
        
        # Structured sections
        "section_headings": SelectorSpec(
            css="h2, h3",
            multiple=True
        ),
        "definition_box": SelectorSpec(
            css=".definition-box, .info-box, .highlight-box",
            multiple=True,
            required=False
        ),
        
        # Math content
        "math_expressions": SelectorSpec(
            css=".MathJax, .MathJax_Display, .katex",
            multiple=True,
            required=False
        ),
        "formula_container": SelectorSpec(
            css=".formula-box, .equation-box",
            multiple=True,
            required=False
        ),
        
        # Examples and problems
        "example_box": SelectorSpec(
            css=".example-box, .worked-example, .solved-example",
            multiple=True,
            required=False
        ),
        "practice_problems": SelectorSpec(
            css=".practice-questions, .exercise-section",
            required=False
        ),
        
        # Tables
        "data_tables": SelectorSpec(
            css="table.data-table, .article-content table",
            multiple=True,
            required=False
        ),
        
        # Images
        "content_images": SelectorSpec(
            css=".article-content img, .post-content img",
            attribute="src",
            multiple=True,
            required=False
        ),
        "image_captions": SelectorSpec(
            css="figcaption, .image-caption",
            multiple=True,
            required=False
        ),
        
        # Navigation
        "related_topics": SelectorSpec(
            css=".related-topics a, .related-posts a",
            attribute="href",
            multiple=True,
            required=False
        ),
    },
    rate_limit=RateLimitSpec(
        min_interval_seconds=3.0,  # More conservative due to Cloudflare
        max_requests_per_minute=15,
        burst_allowed=1,
        respect_retry_after=True
    ),
    retry=RetrySpec(
        max_retries=4,
        backoff_base=3.0,
        backoff_max_seconds=32.0,
        retry_on_status_codes=[429, 500, 502, 503, 504, 403]  # Include 403 for Cloudflare
    ),
    requires_js_rendering=True,
    requires_mathjax_wait=True,
    cloudflare_protected=True,
    robots_txt_notes="Generally permissive, but heavy JS rendering required"
)


# =============================================================================
# VEDANTU SPECIFICATION
# =============================================================================

VEDANTU_SPEC = SourceHandlerSpec(
    name="Vedantu",
    domain="vedantu.com",
    base_urls=[
        "https://www.vedantu.com/physics/",
        "https://www.vedantu.com/chemistry/",
        "https://www.vedantu.com/maths/",
        "https://www.vedantu.com/biology/",
        "https://www.vedantu.com/ncert-solutions/",
    ],
    selectors={
        # Main content
        "article_content": SelectorSpec(
            css=".article-content, .main-content, .topic-content",
            fallback_css="article p, .content-area p",
            multiple=True
        ),
        "page_title": SelectorSpec(
            css="h1.page-title, h1.topic-title",
            fallback_css="h1"
        ),
        
        # NCERT-specific content
        "ncert_solution": SelectorSpec(
            css=".ncert-solution, .solution-content",
            multiple=True,
            required=False
        ),
        "question_answer": SelectorSpec(
            css=".qa-pair, .question-answer-block",
            multiple=True,
            required=False
        ),
        
        # Math content
        "math_expressions": SelectorSpec(
            css=".MathJax, .katex, .math-content",
            multiple=True,
            required=False
        ),
        
        # Key concepts
        "key_points": SelectorSpec(
            css=".key-points, .important-points, .summary-box",
            multiple=True,
            required=False
        ),
        "definition": SelectorSpec(
            css=".definition, .term-definition",
            multiple=True,
            required=False
        ),
        
        # Examples
        "solved_examples": SelectorSpec(
            css=".solved-example, .example-solution",
            multiple=True,
            required=False
        ),
        
        # PDF download (important for Vedantu)
        "pdf_download_link": SelectorSpec(
            css="a[href*='/content-files-downloadable/'], a.pdf-download",
            attribute="href",
            required=False
        ),
        
        # FAQs
        "faq_section": SelectorSpec(
            css=".faq-section, .frequently-asked",
            required=False
        ),
        "faq_questions": SelectorSpec(
            css=".faq-question, .faq-item h3",
            multiple=True,
            required=False
        ),
        "faq_answers": SelectorSpec(
            css=".faq-answer, .faq-item .answer",
            multiple=True,
            required=False
        ),
    },
    rate_limit=RateLimitSpec(
        min_interval_seconds=3.0,
        max_requests_per_minute=15,
        burst_allowed=1
    ),
    retry=RetrySpec(
        max_retries=3,
        backoff_base=2.5,
        backoff_max_seconds=20.0
    ),
    requires_js_rendering=True,
    requires_mathjax_wait=True,
    cloudflare_protected=True,
    has_pdf_downloads=True,
    robots_txt_notes="PDF solutions available at /content-files-downloadable/"
)


# =============================================================================
# YOUTUBE TRANSCRIPT SPECIFICATION
# =============================================================================

YOUTUBE_SPEC = SourceHandlerSpec(
    name="YouTube",
    domain="youtube.com",
    base_urls=[
        "https://www.youtube.com/watch",
        "https://youtu.be/",
    ],
    selectors={
        # Note: YouTube uses youtube-transcript-api, not DOM selectors
        # These selectors are for fallback DOM extraction if API fails
        
        "video_title": SelectorSpec(
            css="h1.ytd-video-primary-info-renderer, #title h1",
            required=False
        ),
        "channel_name": SelectorSpec(
            css="#channel-name a, ytd-channel-name a",
            required=False
        ),
        "description": SelectorSpec(
            css="#description-inline-expander, ytd-text-inline-expander",
            required=False
        ),
        "transcript_cue": SelectorSpec(
            css="ytd-transcript-segment-renderer .segment-text",
            multiple=True,
            required=False
        ),
    },
    rate_limit=RateLimitSpec(
        min_interval_seconds=1.0,  # API is more permissive
        max_requests_per_minute=30,
        burst_allowed=5
    ),
    retry=RetrySpec(
        max_retries=2,
        backoff_base=2.0,
        backoff_max_seconds=8.0,
        retry_on_status_codes=[429, 500, 503]
    ),
    requires_js_rendering=False,  # Uses API, not browser
    robots_txt_notes="Transcript API bypasses robots.txt; use responsibly"
)


# =============================================================================
# GOOGLE SEARCH SPECIFICATION (FALLBACK)
# =============================================================================

GOOGLE_SEARCH_SPEC = SourceHandlerSpec(
    name="Google Search",
    domain="google.com",
    base_urls=[
        "https://www.google.com/search",
    ],
    selectors={
        # Search results
        "search_results": SelectorSpec(
            css="div.g, div[data-hveid]",
            multiple=True
        ),
        "result_title": SelectorSpec(
            css="h3",
            multiple=True
        ),
        "result_link": SelectorSpec(
            css="a[href^='http']",
            attribute="href",
            multiple=True
        ),
        "result_snippet": SelectorSpec(
            css=".VwiC3b, .lEBKkf",
            multiple=True
        ),
        
        # Featured snippet
        "featured_snippet": SelectorSpec(
            css=".xpdopen, .ifM9O",
            required=False
        ),
        
        # Knowledge panel
        "knowledge_panel": SelectorSpec(
            css=".kp-wholepage, .knowledge-panel",
            required=False
        ),
    },
    rate_limit=RateLimitSpec(
        min_interval_seconds=5.0,  # Very conservative for Google
        max_requests_per_minute=10,
        burst_allowed=1
    ),
    retry=RetrySpec(
        max_retries=2,
        backoff_base=5.0,
        backoff_max_seconds=60.0,
        retry_on_status_codes=[429, 503]
    ),
    requires_js_rendering=True,
    cloudflare_protected=False,
    robots_txt_notes="Use as fallback only; prefer educational sites"
)


# =============================================================================
# HANDLER REGISTRY
# =============================================================================

SOURCE_HANDLER_SPECS: dict[str, SourceHandlerSpec] = {
    "khan_academy": KHAN_ACADEMY_SPEC,
    "byjus": BYJUS_SPEC,
    "vedantu": VEDANTU_SPEC,
    "youtube": YOUTUBE_SPEC,
    "google_search": GOOGLE_SEARCH_SPEC,
}


def get_spec_for_url(url: str) -> Optional[SourceHandlerSpec]:
    """Get the appropriate handler spec for a URL."""
    for spec in SOURCE_HANDLER_SPECS.values():
        if spec.domain in url:
            return spec
    return None


def get_spec_by_name(name: str) -> Optional[SourceHandlerSpec]:
    """Get handler spec by name."""
    return SOURCE_HANDLER_SPECS.get(name.lower().replace(" ", "_"))


# =============================================================================
# URL GENERATION TEMPLATES
# =============================================================================

URL_TEMPLATES = {
    "khan_academy": {
        "physics": "https://www.khanacademy.org/science/physics/{topic_slug}",
        "chemistry": "https://www.khanacademy.org/science/chemistry/{topic_slug}",
        "mathematics": "https://www.khanacademy.org/math/{topic_slug}",
        "biology": "https://www.khanacademy.org/science/biology/{topic_slug}",
    },
    "byjus": {
        "physics": "https://byjus.com/physics/{topic_slug}/",
        "chemistry": "https://byjus.com/chemistry/{topic_slug}/",
        "mathematics": "https://byjus.com/maths/{topic_slug}/",
        "biology": "https://byjus.com/biology/{topic_slug}/",
        "cbse": "https://byjus.com/cbse-notes/class-{grade}-{subject}-{topic_slug}/",
    },
    "vedantu": {
        "physics": "https://www.vedantu.com/physics/{topic_slug}",
        "chemistry": "https://www.vedantu.com/chemistry/{topic_slug}",
        "mathematics": "https://www.vedantu.com/maths/{topic_slug}",
        "biology": "https://www.vedantu.com/biology/{topic_slug}",
        "ncert": "https://www.vedantu.com/ncert-solutions/class-{grade}-science-chapter-{topic_slug}",
    },
}


def generate_urls_for_topic(
    source: str,
    subject: str,
    topic: str,
    grade: int = 8
) -> list[str]:
    """
    Generate URLs for a topic from a specific source.
    
    Args:
        source: Source name (khan_academy, byjus, vedantu)
        subject: Subject name
        topic: Topic name
        grade: Student grade (for CBSE/NCERT URLs)
        
    Returns:
        List of URLs to crawl
    """
    topic_slug = topic.lower().replace(" ", "-").replace(",", "").replace("'", "")
    subject_lower = subject.lower()
    
    templates = URL_TEMPLATES.get(source, {})
    urls = []
    
    # Add subject-specific URL
    if subject_lower in templates:
        url = templates[subject_lower].format(
            topic_slug=topic_slug,
            grade=grade,
            subject=subject_lower
        )
        urls.append(url)
    
    # Add CBSE/NCERT URL if available
    if "cbse" in templates:
        url = templates["cbse"].format(
            topic_slug=topic_slug,
            grade=grade,
            subject=subject_lower
        )
        urls.append(url)
    
    if "ncert" in templates:
        url = templates["ncert"].format(
            topic_slug=topic_slug,
            grade=grade
        )
        urls.append(url)
    
    return urls


# Example output
if __name__ == "__main__":
    # Print Khan Academy selectors
    print("Khan Academy Selectors:")
    print("-" * 50)
    for name, selector in KHAN_ACADEMY_SPEC.selectors.items():
        print(f"  {name}: {selector.css}")
    
    print("\n\nGenerated URLs for 'Work Done':")
    print("-" * 50)
    for source in ["khan_academy", "byjus", "vedantu"]:
        urls = generate_urls_for_topic(source, "physics", "Work Done", grade=8)
        print(f"\n{source}:")
        for url in urls:
            print(f"  • {url}")
