"""
Rate limiter for controlling request frequency per domain.

Usage:
    limiter = RateLimiter()
    await limiter.acquire("khanacademy.org")  # Waits if needed
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    min_interval_seconds: float = 2.0
    max_requests_per_minute: int = 20
    burst_allowance: int = 2


@dataclass
class DomainState:
    """Tracks request state for a single domain."""
    last_request_time: float = 0.0
    request_count: int = 0
    minute_start_time: float = field(default_factory=time.time)


class RateLimiter:
    """
    Async rate limiter that enforces per-domain request limits.

    Features:
    - Minimum interval between requests
    - Maximum requests per minute
    - Automatic waiting when limits exceeded
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._domains: Dict[str, DomainState] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str) -> float:
        """
        Acquire permission to make a request to domain.
        Blocks until request is allowed.

        Args:
            domain: The domain to rate limit

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            wait_time = 0.0
            state = self._get_domain_state(domain)
            current_time = time.time()

            # Check if minute needs reset (60s passed since minute start)
            if current_time - state.minute_start_time >= 60.0:
                state.request_count = 0
                state.minute_start_time = current_time

            # Check max_requests_per_minute limit
            if state.request_count >= self.config.max_requests_per_minute:
                # Need to wait until the minute resets
                time_until_reset = 60.0 - (current_time - state.minute_start_time)
                if time_until_reset > 0:
                    await asyncio.sleep(time_until_reset)
                    wait_time += time_until_reset
                    current_time = time.time()
                    # Reset the minute counter
                    state.request_count = 0
                    state.minute_start_time = current_time

            # Check min_interval_seconds between requests
            time_since_last = current_time - state.last_request_time
            if time_since_last < self.config.min_interval_seconds:
                interval_wait = self.config.min_interval_seconds - time_since_last
                await asyncio.sleep(interval_wait)
                wait_time += interval_wait
                current_time = time.time()

            # Update state
            state.last_request_time = current_time
            state.request_count += 1

            return wait_time

    def _get_domain_state(self, domain: str) -> DomainState:
        """Get or create state for domain."""
        if domain not in self._domains:
            self._domains[domain] = DomainState()
        return self._domains[domain]
