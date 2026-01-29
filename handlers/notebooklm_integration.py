"""
EduCrawler NotebookLM Integration
=================================

Automated integration with Google NotebookLM for content compilation.

This module provides:
- Browser automation via Playwright for NotebookLM interaction
- Persistent authentication state management
- Source addition (Website, YouTube, Text)
- Output generation triggering
- Content extraction

Usage:
    from notebooklm_integration import NotebookLMIntegrator
    
    integrator = NotebookLMIntegrator(auth_state_path="nlm_auth.json")
    await integrator.create_notebook("Work Done - Grade 8 Physics")
    await integrator.add_website_source("https://khanacademy.org/...")
    content = await integrator.generate_study_guide()

Prerequisites:
    1. Install Playwright: pip install playwright && playwright install chromium
    2. Manual login once to save auth state (see save_auth_state method)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Note: Playwright import will be available at runtime
# from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NotebookLMConfig:
    """Configuration for NotebookLM integration."""
    
    base_url: str = "https://notebooklm.google.com"
    auth_state_path: str = "notebooklm_auth.json"
    
    # Browser settings
    headless: bool = False  # Must be False for initial auth
    viewport_width: int = 1280
    viewport_height: int = 800
    slow_mo: int = 100  # Milliseconds between actions
    
    # Timeouts (milliseconds)
    page_load_timeout: int = 30000
    element_timeout: int = 10000
    compilation_timeout: int = 60000
    
    # Limits
    max_sources_per_notebook: int = 50
    max_text_source_chars: int = 500000
    
    # Delays between actions (seconds)
    action_delay: float = 1.0
    source_processing_delay: float = 3.0


class SourceChipType(str, Enum):
    """NotebookLM source type chips."""
    WEBSITE = "Website"
    YOUTUBE = "YouTube"
    TEXT = "Copied text"
    GOOGLE_DOCS = "Google Docs"
    GOOGLE_SLIDES = "Google Slides"
    PDF = "PDF"


class OutputType(str, Enum):
    """NotebookLM Studio output types."""
    STUDY_GUIDE = "Study guide"
    BRIEFING_DOC = "Briefing doc"
    FAQ = "FAQ"
    TIMELINE = "Timeline"
    TABLE_OF_CONTENTS = "Table of contents"
    AUDIO_OVERVIEW = "Audio Overview"


# =============================================================================
# DOM SELECTORS (Discovered via Antigravity inspection)
# =============================================================================

@dataclass
class NotebookLMSelectors:
    """CSS selectors for NotebookLM DOM elements."""
    
    # Authentication
    sign_in_button: str = "button:has-text('Sign in')"
    google_account_picker: str = "div[data-identifier]"
    
    # Main navigation
    create_button: str = "button:has-text('Create'), button:has-text('New notebook')"
    notebook_list: str = ".notebook-list, [data-notebook-id]"
    notebook_title_input: str = "input[aria-label*='title'], .notebook-title-input"
    
    # Source panel (left column)
    add_source_button: str = "button:has-text('Add source'), button[aria-label*='Add source']"
    source_type_chip: str = "span.mdc-evolution-chip__text-label"
    source_url_input: str = "input[type='text'][placeholder*='URL'], input[type='url']"
    source_text_input: str = "textarea[placeholder*='Paste'], .text-source-input"
    insert_button: str = "button:has-text('Insert'), button:has-text('Add')"
    source_list: str = ".source-list, [data-source-id]"
    source_item: str = ".source-item, [role='listitem']"
    source_processing_indicator: str = ".processing-indicator, .loading-spinner"
    
    # Chat panel (middle column)
    chat_input: str = "textarea[placeholder*='Ask'], .chat-input"
    send_button: str = "button[aria-label*='Send'], button:has-text('Send')"
    chat_messages: str = ".chat-message, [role='article']"
    
    # Studio panel (right column)
    studio_tab: str = "button:has-text('Studio'), [role='tab']:has-text('Studio')"
    output_type_button: str = "button:has-text('{output_type}')"
    generate_button: str = "button:has-text('Generate'), button:has-text('Create')"
    generated_content: str = ".generated-content, .studio-output, [data-output-content]"
    copy_button: str = "button[aria-label*='Copy'], button:has-text('Copy')"
    
    # Dialogs
    dialog_container: str = "[role='dialog'], .modal, .dialog"
    dialog_close_button: str = "button[aria-label*='Close'], button:has-text('Cancel')"
    confirm_button: str = "button:has-text('Confirm'), button:has-text('OK')"
    
    # Error indicators
    error_message: str = ".error-message, [role='alert']"
    toast_notification: str = ".toast, .snackbar, [role='status']"


SELECTORS = NotebookLMSelectors()


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class NotebookSource:
    """Represents a source added to a notebook."""
    
    source_type: SourceChipType
    identifier: str  # URL or title for text
    status: str = "pending"  # pending, processing, ready, failed
    added_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


@dataclass
class NotebookSession:
    """Represents an active NotebookLM session."""
    
    notebook_id: Optional[str] = None
    notebook_title: str = ""
    sources: list[NotebookSource] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def source_count(self) -> int:
        return len(self.sources)
    
    @property
    def ready_sources(self) -> int:
        return sum(1 for s in self.sources if s.status == "ready")


@dataclass
class GeneratedOutput:
    """Output generated by NotebookLM Studio."""
    
    output_type: OutputType
    content: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    source_count: int = 0


# =============================================================================
# NOTEBOOKLM INTEGRATOR
# =============================================================================

class NotebookLMIntegrator:
    """
    Automated integration with Google NotebookLM.
    
    This class provides browser automation for:
    - Creating notebooks
    - Adding sources (websites, YouTube, text)
    - Generating outputs (study guides, FAQs, etc.)
    - Extracting generated content
    
    Usage:
        integrator = NotebookLMIntegrator()
        
        # First time: save authentication state
        await integrator.save_auth_state()
        
        # Subsequent uses: automated workflow
        await integrator.initialize()
        await integrator.create_notebook("My Topic")
        await integrator.add_website_source("https://example.com")
        content = await integrator.generate_study_guide()
        await integrator.close()
    """
    
    def __init__(self, config: Optional[NotebookLMConfig] = None):
        self.config = config or NotebookLMConfig()
        self.logger = logging.getLogger(__name__)
        
        # Browser state (set during initialize)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        
        # Session state
        self.session: Optional[NotebookSession] = None
    
    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================
    
    async def initialize(self) -> None:
        """
        Initialize browser with saved authentication state.
        
        Raises:
            FileNotFoundError: If auth state file doesn't exist
        """
        from playwright.async_api import async_playwright
        
        auth_path = Path(self.config.auth_state_path)
        if not auth_path.exists():
            raise FileNotFoundError(
                f"Auth state file not found: {auth_path}\n"
                f"Run save_auth_state() first to create it."
            )
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )
        self._context = await self._browser.new_context(
            storage_state=str(auth_path),
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            }
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.config.element_timeout)
        
        # Navigate to NotebookLM
        await self._page.goto(
            self.config.base_url,
            wait_until="networkidle",
            timeout=self.config.page_load_timeout
        )
        
        self.logger.info("NotebookLM integrator initialized")
    
    async def close(self) -> None:
        """Close browser and cleanup resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        
        self.logger.info("NotebookLM integrator closed")
    
    async def save_auth_state(self) -> None:
        """
        Interactive method to save authentication state.
        
        Opens browser for manual Google login, then saves the
        authentication state for future automated use.
        
        Usage:
            integrator = NotebookLMIntegrator()
            await integrator.save_auth_state()
            # Browser opens, log in manually
            # Press Enter in terminal when done
        """
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=False,  # Must be visible for manual login
            slow_mo=100
        )
        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            }
        )
        self._page = await self._context.new_page()
        
        # Navigate to NotebookLM
        await self._page.goto(self.config.base_url)
        
        print("\n" + "=" * 60)
        print("MANUAL LOGIN REQUIRED")
        print("=" * 60)
        print("1. Log in to your Google account in the browser window")
        print("2. Wait for NotebookLM to fully load")
        print("3. Press ENTER in this terminal when ready")
        print("=" * 60 + "\n")
        
        input("Press ENTER after logging in...")
        
        # Save authentication state
        await self._context.storage_state(path=self.config.auth_state_path)
        
        print(f"\n✓ Authentication state saved to: {self.config.auth_state_path}")
        
        await self.close()
    
    # =========================================================================
    # NOTEBOOK MANAGEMENT
    # =========================================================================
    
    async def create_notebook(self, title: str) -> str:
        """
        Create a new notebook.
        
        Args:
            title: Notebook title
            
        Returns:
            Notebook ID (extracted from URL)
        """
        self._ensure_initialized()
        
        # Click create button
        await self._click(SELECTORS.create_button)
        await asyncio.sleep(self.config.action_delay)
        
        # Wait for new notebook to load
        await self._page.wait_for_load_state("networkidle")
        
        # Set title if input is available
        try:
            title_input = await self._page.wait_for_selector(
                SELECTORS.notebook_title_input,
                timeout=5000
            )
            if title_input:
                await title_input.fill(title)
                await title_input.press("Enter")
        except Exception as e:
            self.logger.warning(f"Could not set notebook title: {e}")
        
        # Extract notebook ID from URL
        notebook_id = self._extract_notebook_id()
        
        # Initialize session
        self.session = NotebookSession(
            notebook_id=notebook_id,
            notebook_title=title
        )
        
        self.logger.info(f"Created notebook: {title} (ID: {notebook_id})")
        return notebook_id
    
    def _extract_notebook_id(self) -> Optional[str]:
        """Extract notebook ID from current URL."""
        url = self._page.url
        # NotebookLM URLs typically look like:
        # https://notebooklm.google.com/notebook/NOTEBOOK_ID
        if "/notebook/" in url:
            return url.split("/notebook/")[-1].split("?")[0]
        return None
    
    # =========================================================================
    # SOURCE MANAGEMENT
    # =========================================================================
    
    async def add_website_source(self, url: str) -> bool:
        """
        Add a website URL as a source.
        
        Args:
            url: Website URL to add
            
        Returns:
            True if source was added successfully
        """
        return await self._add_source(SourceChipType.WEBSITE, url)
    
    async def add_youtube_source(self, video_url: str) -> bool:
        """
        Add a YouTube video as a source.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            True if source was added successfully
        """
        return await self._add_source(SourceChipType.YOUTUBE, video_url)
    
    async def add_text_source(self, title: str, content: str) -> bool:
        """
        Add pasted text as a source.
        
        Args:
            title: Title for the text source
            content: Text content to add
            
        Returns:
            True if source was added successfully
        """
        self._ensure_initialized()
        
        if len(content) > self.config.max_text_source_chars:
            self.logger.warning(
                f"Text content exceeds limit ({len(content)} > "
                f"{self.config.max_text_source_chars}). Truncating."
            )
            content = content[:self.config.max_text_source_chars]
        
        # Click add source
        await self._click(SELECTORS.add_source_button)
        await asyncio.sleep(self.config.action_delay)
        
        # Select "Copied text" chip
        await self._click_chip(SourceChipType.TEXT)
        await asyncio.sleep(self.config.action_delay)
        
        # Fill text area
        text_input = await self._page.wait_for_selector(SELECTORS.source_text_input)
        await text_input.fill(content)
        
        # Click insert
        await self._click(SELECTORS.insert_button)
        
        # Wait for processing
        await self._wait_for_source_processing()
        
        # Track source
        source = NotebookSource(
            source_type=SourceChipType.TEXT,
            identifier=title,
            status="ready"
        )
        if self.session:
            self.session.sources.append(source)
        
        self.logger.info(f"Added text source: {title}")
        return True
    
    async def _add_source(self, source_type: SourceChipType, url: str) -> bool:
        """Internal method to add URL-based sources."""
        self._ensure_initialized()
        
        try:
            # Click add source button
            await self._click(SELECTORS.add_source_button)
            await asyncio.sleep(self.config.action_delay)
            
            # Select source type chip
            await self._click_chip(source_type)
            await asyncio.sleep(self.config.action_delay)
            
            # Fill URL
            url_input = await self._page.wait_for_selector(SELECTORS.source_url_input)
            await url_input.fill(url)
            
            # Click insert
            await self._click(SELECTORS.insert_button)
            
            # Wait for processing
            await self._wait_for_source_processing()
            
            # Track source
            source = NotebookSource(
                source_type=source_type,
                identifier=url,
                status="ready"
            )
            if self.session:
                self.session.sources.append(source)
            
            self.logger.info(f"Added {source_type.value} source: {url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add source: {e}")
            if self.session:
                source = NotebookSource(
                    source_type=source_type,
                    identifier=url,
                    status="failed",
                    error_message=str(e)
                )
                self.session.sources.append(source)
            return False
    
    async def _click_chip(self, source_type: SourceChipType) -> None:
        """Click on a source type chip."""
        chip_selector = f"{SELECTORS.source_type_chip}:has-text('{source_type.value}')"
        chip = await self._page.wait_for_selector(chip_selector)
        await chip.click()
    
    async def _wait_for_source_processing(self) -> None:
        """Wait for source to finish processing."""
        try:
            # Wait for processing indicator to appear
            await self._page.wait_for_selector(
                SELECTORS.source_processing_indicator,
                timeout=2000
            )
            # Wait for it to disappear
            await self._page.wait_for_selector(
                SELECTORS.source_processing_indicator,
                state="hidden",
                timeout=self.config.compilation_timeout
            )
        except Exception:
            # Indicator might not appear for fast processing
            pass
        
        # Additional delay for safety
        await asyncio.sleep(self.config.source_processing_delay)
    
    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================
    
    async def generate_study_guide(self) -> GeneratedOutput:
        """Generate a study guide from sources."""
        return await self._generate_output(OutputType.STUDY_GUIDE)
    
    async def generate_faq(self) -> GeneratedOutput:
        """Generate FAQ from sources."""
        return await self._generate_output(OutputType.FAQ)
    
    async def generate_briefing_doc(self) -> GeneratedOutput:
        """Generate briefing document from sources."""
        return await self._generate_output(OutputType.BRIEFING_DOC)
    
    async def generate_timeline(self) -> GeneratedOutput:
        """Generate timeline from sources."""
        return await self._generate_output(OutputType.TIMELINE)
    
    async def _generate_output(self, output_type: OutputType) -> GeneratedOutput:
        """Internal method to generate any output type."""
        self._ensure_initialized()
        
        # Click Studio tab if not already active
        try:
            studio_tab = await self._page.wait_for_selector(
                SELECTORS.studio_tab,
                timeout=3000
            )
            await studio_tab.click()
            await asyncio.sleep(self.config.action_delay)
        except Exception:
            pass  # Tab might already be active
        
        # Click the output type button
        output_button_selector = SELECTORS.output_type_button.format(
            output_type=output_type.value
        )
        await self._click(output_button_selector)
        await asyncio.sleep(self.config.action_delay)
        
        # Click generate if needed
        try:
            generate_button = await self._page.wait_for_selector(
                SELECTORS.generate_button,
                timeout=3000
            )
            await generate_button.click()
        except Exception:
            pass  # Generation might start automatically
        
        # Wait for content to appear
        content_element = await self._page.wait_for_selector(
            SELECTORS.generated_content,
            timeout=self.config.compilation_timeout
        )
        
        # Extract content
        content = await content_element.text_content()
        
        output = GeneratedOutput(
            output_type=output_type,
            content=content or "",
            source_count=self.session.source_count if self.session else 0
        )
        
        self.logger.info(f"Generated {output_type.value} ({len(content or '')} chars)")
        return output
    
    async def ask_question(self, question: str) -> str:
        """
        Ask a question about the sources via chat.
        
        Args:
            question: Question to ask
            
        Returns:
            Response text
        """
        self._ensure_initialized()
        
        # Fill chat input
        chat_input = await self._page.wait_for_selector(SELECTORS.chat_input)
        await chat_input.fill(question)
        
        # Send
        await self._click(SELECTORS.send_button)
        
        # Wait for response
        await asyncio.sleep(3)  # Wait for processing
        
        # Get latest message
        messages = await self._page.query_selector_all(SELECTORS.chat_messages)
        if messages:
            last_message = messages[-1]
            return await last_message.text_content() or ""
        
        return ""
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _ensure_initialized(self) -> None:
        """Ensure browser is initialized."""
        if self._page is None:
            raise RuntimeError(
                "NotebookLM integrator not initialized. "
                "Call initialize() first."
            )
    
    async def _click(self, selector: str) -> None:
        """Click an element with error handling."""
        element = await self._page.wait_for_selector(selector)
        await element.click()
    
    async def _get_text(self, selector: str) -> str:
        """Get text content of an element."""
        element = await self._page.wait_for_selector(selector)
        return await element.text_content() or ""
    
    async def take_screenshot(self, path: str) -> None:
        """Take a screenshot for debugging."""
        if self._page:
            await self._page.screenshot(path=path)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def example_workflow():
    """Example workflow demonstrating NotebookLM integration."""
    
    integrator = NotebookLMIntegrator(
        config=NotebookLMConfig(
            headless=False,
            slow_mo=100
        )
    )
    
    try:
        # Initialize with saved auth
        await integrator.initialize()
        
        # Create notebook
        await integrator.create_notebook("Work Done - Grade 8 Physics")
        
        # Add sources
        await integrator.add_website_source(
            "https://www.khanacademy.org/science/physics/work-and-energy"
        )
        await integrator.add_youtube_source(
            "https://www.youtube.com/watch?v=example"
        )
        await integrator.add_text_source(
            "Additional Notes",
            "Work is defined as force times displacement..."
        )
        
        # Generate study guide
        output = await integrator.generate_study_guide()
        print(f"Generated study guide ({len(output.content)} characters)")
        print(output.content[:500])
        
    finally:
        await integrator.close()


if __name__ == "__main__":
    # To save auth state (run once):
    # asyncio.run(NotebookLMIntegrator().save_auth_state())
    
    # To run example workflow:
    # asyncio.run(example_workflow())
    
    print("NotebookLM Integration Module")
    print("=" * 50)
    print("\nFirst-time setup:")
    print("  python -c \"import asyncio; from notebooklm_integration import NotebookLMIntegrator; asyncio.run(NotebookLMIntegrator().save_auth_state())\"")
    print("\nThen use NotebookLMIntegrator in your code.")
