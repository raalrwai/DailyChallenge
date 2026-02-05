import asyncio
import random
import time
import argparse
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field

# ============================================================
# Configuration Constants
# ============================================================
# These are intentionally centralized to avoid magic numbers
# and make tuning behavior easy without touching logic.

MAX_RETRIES = 3            # Maximum retry attempts per request
BASE_BACKOFF = 0.2         # Initial backoff (seconds) for retries
CACHE_TTL = 5.0            # Time-to-live for cached responses (seconds)
REQUEST_TIMEOUT = 1.0      # Per-request timeout (seconds)


# ============================================================
# Time Utility
# ============================================================
def now() -> float:
    """
    Returns a monotonic clock reading.

    Why monotonic?
    - It is immune to system clock changes (NTP, daylight savings).
    - Safe for measuring durations and latency.
    """
    return time.monotonic()


# ============================================================
# Async Retry Utility
# ============================================================
async def async_retry(
    fn: Callable,
    *,
    retries: int = MAX_RETRIES,
    timeout: float = REQUEST_TIMEOUT,
) -> Any:
    """
    Executes an async function with:
    - timeout enforcement
    - retry logic
    - exponential backoff

    Design notes:
    - The function `fn` is passed as a callable to delay execution.
    - asyncio.wait_for enforces hard timeouts.
    - Backoff helps reduce pressure on unstable downstream systems.
    """
    attempt = 0

    while True:
        try:
            # Enforce timeout per attempt
            return await asyncio.wait_for(fn(), timeout=timeout)

        except Exception:
            attempt += 1

            # Exceeded retry budget → propagate failure
            if attempt > retries:
                raise

            # Exponential backoff: 0.2s, 0.4s, 0.8s, ...
            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)


# ============================================================
# Cache Implementation (TTL-based)
# ============================================================
@dataclass
class CacheEntry:
    """
    Represents a single cached value with an expiration timestamp.
    """
    value: Any
    expires_at: float


class TTLCache:
    """
    Simple in-memory TTL cache.

    Characteristics:
    - No background cleanup thread
    - Lazy eviction on access
    - O(1) lookup
    - Not thread-safe (intentional for simplicity)
    """

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value if present and not expired.
        """
        entry = self._store.get(key)

        # Miss or expired → evict and return None
        if not entry or entry.expires_at < now():
            self._store.pop(key, None)
            return None

        return entry.value

    def set(self, key: str, value: Any):
        """
        Store a value with TTL-based expiration.
        """
        self._store[key] = CacheEntry(
            value=value,
            expires_at=now() + self.ttl
        )


# ============================================================
# Simulated External APIs
# ============================================================
async def fake_api(name: str) -> Dict[str, Any]:
    """
    Simulates an unreliable external API.

    Behavior:
    - Random latency (network jitter)
    - Random failure (~25%)
    - Returns structured payload

    This models real-world services:
    - Partial outages
    - Slow responses
    - Non-deterministic behavior
    """
    await asyncio.sleep(random.uniform(0.05, 0.8))

    # Inject probabilistic failure
    if random.random() < 0.25:
        raise RuntimeError(f"{name} failed")

    return {
        "source": name,
        "value": random.randint(1, 100),
        "timestamp": time.time(),
    }


# ============================================================
# Metrics Collection
# ============================================================
@dataclass
class Metrics:
    """
    Lightweight metrics container.

    Note:
    - Latency stored as raw samples
    - Percentiles can be computed later
    """
    success: int = 0
    failure: int = 0
    latency: list = field(default_factory=list)


# ============================================================
# Aggregator Core
# ============================================================
class Aggregator:
    """
    Coordinates:
    - Cache access
    - Retry logic
    - Concurrent fetching
    - Metrics tracking
    """

    def __init__(self):
        self.cache = TTLCache(CACHE_TTL)
        self.metrics = Metrics()

    async def fetch(self, source: str) -> Dict[str, Any]:
        """
        Fetch data for a single source with:
        - Cache lookup
        - Retry + timeout
        - Metrics instrumentation
        """
        # Fast path: cache hit
        cached = self.cache.get(source)
        if cached:
            return cached

        start = now()

        try:
            # Retry-protected API call
            result = await async_retry(lambda: fake_api(source))

            # Cache successful responses only
            self.cache.set(source, result)

            self.metrics.success += 1
            return result

        except Exception:
            # Failures are tracked but isolated
            self.metrics.failure += 1
            raise

        finally:
            # Latency is tracked regardless of outcome
            self.metrics.latency.append(now() - start)

    async def run(self, sources: list[str]) -> Dict[str, Any]:
        """
        Fetch multiple sources concurrently.

        Design choices:
        - asyncio.gather for parallelism
        - return_exceptions=True prevents cascading failure
        """
        results = {}

        tasks = [self.fetch(src) for src in sources]

        completed = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

        # Normalize output so callers don't deal with exceptions
        for src, result in zip(sources, completed):
            if isinstance(result, Exception):
                results[src] = {"error": str(result)}
            else:
                results[src] = result

        return results


# ============================================================
# CLI Argument Parsing
# ============================================================
def parse_args():
    """
    Parses CLI arguments.

    Example:
    python main.py --sources users orders payments
    """
    parser = argparse.ArgumentParser(
        description="Async Data Aggregator"
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["users", "orders", "payments", "inventory"],
        help="Data sources to fetch"
    )

    return parser.parse_args()


# ============================================================
# Program Entry Point
# ============================================================
async def main():
    """
    Main async entry point.

    Responsibilities:
    - Parse CLI args
    - Execute aggregation
    - Display results + metrics
    """
    args = parse_args()
    aggregator = Aggregator()

    results = await aggregator.run(args.sources)

    print("\nResults:")
    for source, payload in results.items():
        print(f"{source}: {payload}")

    print("\nMetrics:")
    print(f"Success: {aggregator.metrics.success}")
    print(f"Failure: {aggregator.metrics.failure}")

    if aggregator.metrics.latency:
        avg_latency = (
            sum(aggregator.metrics.latency)
            / len(aggregator.metrics.latency)
        )
        print(f"Avg latency: {avg_latency:.3f}s")


# Standard asyncio bootstrap
if __name__ == "__main__":
    asyncio.run(main())
