"""
EduCrawler Base Handler
=======================

Abstract base class for all source handlers.

All source-specific handlers (Khan Academy, Byjus, Vedantu, YouTube, Google)
must inherit from this class and implement the required methods.

Usage:
    from handlers.base import BaseSourceHandler
    
    class KhanAcademyHandler(BaseSourceHandler):
        async def can_handle(self, url: str) -> bool:
            return "khanacademy.org" in url
        
        async def fetch(self, url: str) -> str:
            # Implementation
            pass
        
        async def extract(self, html: str, url: str) -> ExtractedContent:
            # Implementation
            pass
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypeVar

from pydantic import BaseModel


# Type variable for generic handler
T = TypeVar("T", bound="BaseSourceHandler")


# =============================================================================
# HANDLER CONFIGURATION
# =============================================================================

@dataclass
class HandlerConfig:
    """Configuration for source handlers."""
    
    # Rate limiting
    min_request_interval: float = 2.0
    max_requests_per_minute: int = 20
    
    # Timeouts
    page_load_timeout: int = 30000  # milliseconds
    element_timeout: int = 10000
    
    # Retry
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 32.0
    
    # Browser
    requires_js: bool = True
    wait_for_mathjax: bool = True
    mathjax_wait_seconds: float = 3.0
    
    # Quality
    min_content_length: int = 200
    min_quality_score: float = 0.6


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    
    success: bool
    html: str = ""
    status_code: int = 0
    content_type: str = ""
    
    # Timing
    fetch_duration_seconds: float = 0.0
    page_load_time_seconds: float = 0.0
    
    # Error info
    error: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    final_url: str = ""  # After redirects
    response_headers: dict = field(default_factory=dict)


# =============================================================================
# ABSTRACT BASE HANDLER
# =============================================================================

class BaseSourceHandler(ABC):
    """
    Abstract base class for all source handlers.
    
    Each source handler must implement:
    - can_handle(): Check if handler can process a URL
    - fetch(): Retrieve HTML content from URL
    - extract(): Parse HTML and extract structured content
    - validate(): Validate extracted content quality
    
    Optional overrides:
    - get_urls_for_topic(): Generate URLs for a topic
    - pre_fetch_hook(): Run before fetch
    - post_fetch_hook(): Run after fetch
    """
    
    # Class attributes (override in subclass)
    name: str = "base"
    domain: str = ""
    priority: int = 10
    
    def __init__(self, config: Optional[HandlerConfig] = None):
        """
        Initialize handler with optional configuration.
        
        Args:
            config: Handler configuration (uses defaults if None)
        """
        self.config = config or HandlerConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Rate limiting state
        self._last_request_time: Optional[datetime] = None
        self._request_count: int = 0
        self._minute_start: Optional[datetime] = None
        
        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_open_time: Optional[datetime] = None
    
    # =========================================================================
    # ABSTRACT METHODS (MUST IMPLEMENT)
    # =========================================================================
    
    @abstractmethod
    async def can_handle(self, url: str) -> bool:
        """
        Check if this handler can process the given URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if handler can process this URL
        """
        pass
    
    @abstractmethod
    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch HTML content from URL.
        
        Implementations should:
        - Handle rate limiting
        - Handle retries with backoff
        - Wait for JS rendering if required
        - Wait for MathJax if required
        
        Args:
            url: URL to fetch
            
        Returns:
            FetchResult with HTML content or error
        """
        pass
    
    @abstractmethod
    async def extract(self, html: str, url: str) -> Any:
        """
        Extract structured content from HTML.
        
        Implementations should:
        - Parse HTML using configured selectors
        - Extract concepts, formulas, examples
        - Preserve LaTeX/MathML
        - Handle missing elements gracefully
        
        Args:
            html: HTML content to parse
            url: Source URL (for metadata)
            
        Returns:
            ExtractedContent model with structured data
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS (CAN OVERRIDE)
    # =========================================================================
    
    async def validate(self, content: Any) -> bool:
        """
        Validate extracted content meets quality thresholds.
        
        Default implementation checks:
        - Content has minimum items
        - Quality score meets threshold
        
        Args:
            content: ExtractedContent to validate
            
        Returns:
            True if content is valid
        """
        # Default validation - override in subclass for specific checks
        if hasattr(content, "quality_score"):
            return content.quality_score >= self.config.min_quality_score
        if hasattr(content, "total_content_items"):
            return content.total_content_items > 0
        return True
    
    async def get_urls_for_topic(
        self,
        subject: str,
        topic: str,
        grade: int = 8
    ) -> list[str]:
        """
        Generate URLs to crawl for a given topic.
        
        Override in subclass to implement source-specific URL generation.
        
        Args:
            subject: Subject name (physics, chemistry, etc.)
            topic: Topic name
            grade: Student grade level
            
        Returns:
            List of URLs to crawl
        """
        return []
    
    async def pre_fetch_hook(self, url: str) -> None:
        """
        Hook called before fetch operation.
        
        Override for custom pre-fetch logic (e.g., session setup).
        
        Args:
            url: URL about to be fetched
        """
        pass
    
    async def post_fetch_hook(self, url: str, result: FetchResult) -> None:
        """
        Hook called after fetch operation.
        
        Override for custom post-fetch logic (e.g., logging, metrics).
        
        Args:
            url: URL that was fetched
            result: Fetch result
        """
        pass
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    async def wait_for_rate_limit(self) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Implements:
        - Minimum interval between requests
        - Maximum requests per minute
        """
        now = datetime.utcnow()
        
        # Check minimum interval
        if self._last_request_time:
            elapsed = (now - self._last_request_time).total_seconds()
            if elapsed < self.config.min_request_interval:
                wait_time = self.config.min_request_interval - elapsed
                self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Check requests per minute
        if self._minute_start:
            minute_elapsed = (now - self._minute_start).total_seconds()
            if minute_elapsed < 60:
                if self._request_count >= self.config.max_requests_per_minute:
                    wait_time = 60 - minute_elapsed
                    self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s (per-minute limit)")
                    await asyncio.sleep(wait_time)
                    self._request_count = 0
                    self._minute_start = datetime.utcnow()
            else:
                # Reset minute counter
                self._request_count = 0
                self._minute_start = datetime.utcnow()
        else:
            self._minute_start = datetime.utcnow()
        
        self._last_request_time = datetime.utcnow()
        self._request_count += 1
    
    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================
    
    def record_success(self) -> None:
        """Record successful request for circuit breaker."""
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            self.logger.info(f"Circuit breaker closed for {self.name}")
    
    def record_failure(self) -> None:
        """Record failed request for circuit breaker."""
        self._consecutive_failures += 1
        
        # Open circuit after 3 consecutive failures
        if self._consecutive_failures >= 3 and not self._circuit_open:
            self._circuit_open = True
            self._circuit_open_time = datetime.utcnow()
            self.logger.warning(f"Circuit breaker OPENED for {self.name}")
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_open:
            return False
        
        # Half-open after 5 minutes
        if self._circuit_open_time:
            elapsed = (datetime.utcnow() - self._circuit_open_time).total_seconds()
            if elapsed > 300:  # 5 minutes
                self.logger.info(f"Circuit breaker half-open for {self.name}")
                return False
        
        return True
    
    # =========================================================================
    # RETRY LOGIC
    # =========================================================================
    
    async def fetch_with_retry(self, url: str) -> FetchResult:
        """
        Fetch with automatic retry on failure.
        
        Implements exponential backoff with jitter.
        
        Args:
            url: URL to fetch
            
        Returns:
            FetchResult (may contain error if all retries failed)
        """
        import random
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            if self.is_circuit_open():
                return FetchResult(
                    success=False,
                    error="Circuit breaker open"
                )
            
            await self.wait_for_rate_limit()
            await self.pre_fetch_hook(url)
            
            try:
                result = await self.fetch(url)
                await self.post_fetch_hook(url, result)
                
                if result.success:
                    self.record_success()
                    return result
                
                last_error = result.error
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
            
            self.record_failure()
            
            # Calculate backoff with jitter
            if attempt < self.config.max_retries - 1:
                backoff = min(
                    self.config.backoff_base ** attempt,
                    self.config.backoff_max
                )
                jitter = random.uniform(0, 0.5) * backoff
                wait_time = backoff + jitter
                
                self.logger.debug(f"Retrying in {wait_time:.2f}s (attempt {attempt + 2})")
                await asyncio.sleep(wait_time)
        
        return FetchResult(
            success=False,
            error=f"All {self.config.max_retries} retries failed. Last error: {last_error}",
            retry_count=self.config.max_retries
        )
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    async def process(self, url: str) -> Any:
        """
        Main entry point: fetch, extract, and validate content.
        
        Args:
            url: URL to process
            
        Returns:
            ExtractedContent if successful, None if failed
            
        Raises:
            ValueError: If handler cannot process URL
        """
        if not await self.can_handle(url):
            raise ValueError(f"Handler {self.name} cannot process URL: {url}")
        
        # Fetch
        result = await self.fetch_with_retry(url)
        if not result.success:
            self.logger.error(f"Failed to fetch {url}: {result.error}")
            return None
        
        # Extract
        try:
            content = await self.extract(result.html, url)
        except Exception as e:
            self.logger.error(f"Failed to extract from {url}: {e}")
            return None
        
        # Validate
        if not await self.validate(content):
            self.logger.warning(f"Content validation failed for {url}")
            return None
        
        return content


# =============================================================================
# HANDLER REGISTRY
# =============================================================================

class HandlerRegistry:
    """
    Registry for source handlers.
    
    Allows registration and lookup of handlers by name or URL.
    
    Usage:
        registry = HandlerRegistry()
        registry.register(KhanAcademyHandler())
        registry.register(ByjusHandler())
        
        handler = registry.get_handler_for_url("https://khanacademy.org/...")
    """
    
    def __init__(self):
        self._handlers: dict[str, BaseSourceHandler] = {}
        self._priority_order: list[str] = []
    
    def register(self, handler: BaseSourceHandler) -> None:
        """Register a handler."""
        self._handlers[handler.name] = handler
        self._priority_order.append(handler.name)
        self._priority_order.sort(key=lambda n: self._handlers[n].priority)
    
    def get_handler(self, name: str) -> Optional[BaseSourceHandler]:
        """Get handler by name."""
        return self._handlers.get(name)
    
    async def get_handler_for_url(self, url: str) -> Optional[BaseSourceHandler]:
        """Get appropriate handler for URL."""
        for name in self._priority_order:
            handler = self._handlers[name]
            if await handler.can_handle(url):
                return handler
        return None
    
    def get_handlers_by_priority(self) -> list[BaseSourceHandler]:
        """Get all handlers ordered by priority."""
        return [self._handlers[name] for name in self._priority_order]
    
    def list_handlers(self) -> list[str]:
        """List all registered handler names."""
        return list(self._handlers.keys())


# Global registry instance
_registry = HandlerRegistry()


def get_registry() -> HandlerRegistry:
    """Get the global handler registry."""
    return _registry


def register_handler(handler: BaseSourceHandler) -> None:
    """Register a handler in the global registry."""
    _registry.register(handler)


# =============================================================================
# HANDLER INTERFACE TYPES
# =============================================================================

# Type alias for handler factory
HandlerFactory = type[BaseSourceHandler]


# Example of handler interface contract
class SourceHandlerInterface:
    """
    Interface documentation for source handlers.
    
    All handlers must implement these methods with the specified signatures.
    """
    
    # Required class attributes
    name: str  # Unique handler name (e.g., "khan_academy")
    domain: str  # Primary domain (e.g., "khanacademy.org")
    priority: int  # Priority order (1 = highest)
    
    # Required methods
    async def can_handle(self, url: str) -> bool: ...
    async def fetch(self, url: str) -> FetchResult: ...
    async def extract(self, html: str, url: str) -> Any: ...
    
    # Optional methods
    async def validate(self, content: Any) -> bool: ...
    async def get_urls_for_topic(self, subject: str, topic: str, grade: int) -> list[str]: ...
