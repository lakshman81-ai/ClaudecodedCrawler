"""
Retry decorator with exponential backoff.

Usage:
    @retry_with_backoff(max_retries=3)
    async def fetch_data():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 32.0
    jitter_range: Tuple[float, float] = (0.0, 0.5)
    retry_on: Tuple[Type[Exception], ...] = (Exception,)


def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(config.backoff_base ** attempt, config.backoff_max)
    jitter = random.uniform(*config.jitter_range)
    return delay + jitter


def retry_with_backoff(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for async functions with retry logic.

    Args:
        max_retries: Maximum retry attempts
        backoff_base: Base for exponential backoff
        retry_on: Exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                backoff_base=backoff_base,
                retry_on=retry_on
            )
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = calculate_backoff(attempt, config)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            raise last_exception
        return wrapper
    return decorator
