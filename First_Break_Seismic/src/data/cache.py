"""
LRU cache management for chunked dataset data with level-based telemetry.

The cache stores chunk-related data in an ``OrderedDict`` and evicts the
least recently used entry when the configured capacity is exceeded.

Telemetry levels:
    INFO:
        Cache evictions and full cache clears.
    DEBUG:
        Cache get/put operations, hits, misses, and state/statistics details.
"""

from collections import OrderedDict
from typing import Any

import torch
from loguru import logger

class LRUCache:
    """
    Least Recently Used (LRU) cache for chunked dataset data.

    Entries are stored in insertion/access order using ``OrderedDict``.
    Accessing an existing key moves it to the end of the cache, marking it
    as the most recently used entry. When the cache reaches ``max_size``,
    the least recently used entry is evicted before inserting a new one.

    Each cache entry is expected to be a dictionary that may contain tensors
    under the ``"data"`` and ``"mask"`` keys. These values are explicitly
    removed during eviction to help release references to potentially large
    tensors. CUDA cache cleanup is also triggered when CUDA is available.

    The cache maintains hit/miss statistics that can be retrieved through
    :meth:`get_stats`.

    Logging:
        INFO:
            Cache evictions and cache clears.
        DEBUG:
            GET/PUT operations, hits, misses, cache state, and statistics.

    Args:
        max_size: Maximum number of entries allowed in the cache. Defaults
            to 3.

    Attributes:
        max_size: Maximum cache capacity.
        cache: Ordered mapping of cache keys to cached chunk data.
        hits: Number of successful cache lookups.
        misses: Number of unsuccessful cache lookups.
    """

    def __init__(self, max_size: int = 3):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries that can be stored before
                the least recently used entry is evicted.
        """
        self.max_size = max_size
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> dict[str, Any] | None:
        """
        Retrieve an entry from the cache.

        If the key exists, the entry is marked as most recently used and
        the cache hit counter is incremented. If the key does not exist,
        the cache miss counter is incremented.

        Args:
            key: Integer identifier of the cached chunk.

        Returns:
            The cached chunk dictionary if ``key`` exists; otherwise ``None``.
        """
        logger.debug(f"[Cache] GET key={key} | active_keys={list(self.cache.keys())}")

        if key not in self.cache:
            self.misses += 1
            logger.debug(f"[Cache] MISS key={key} | misses={self.misses}")
            return None

        self.hits += 1
        self.cache.move_to_end(key)
        logger.debug(f"[Cache] HIT key={key} | hits={self.hits}")
        return self.cache[key]

    def put(self, key: int, value: dict[str, Any]):
        """
        Insert or update an entry in the cache.

        If the key already exists, its value is replaced and the entry is
        marked as most recently used. Otherwise, the new entry is added to
        the cache. If the cache is already at capacity, the least recently
        used entry is evicted first.

        Args:
            key: Integer identifier of the chunk.
            value: Cached chunk data. The value is expected to be a dictionary
                and may contain ``"data"`` and ``"mask"`` entries.
        """
        logger.debug(
            f"[Cache] PUT key={key} | size={len(self.cache)}/{self.max_size} | active_keys={list(self.cache.keys())}"
        )

        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return

        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
            logger.info(
                f"[Cache] EVICT key={oldest_key} | size={len(self.cache)}/{self.max_size} (full)"
            )

        self.cache[key] = value
        logger.debug(f"[Cache] ADD key={key} | new_keys={list(self.cache.keys())}")

    def _evict(self, key: int):
        """
        Remove an entry from the cache and release associated memory.

        The method removes references to ``"data"`` and ``"mask"`` from the
        cached value when present. If CUDA is available, PyTorch's CUDA
        allocator cache is also cleared.

        Args:
            key: Integer identifier of the entry to evict.
        """
        if key in self.cache:
            value = self.cache[key]
            if "data" in value:
                del value["data"]
            if "mask" in value:
                del value["mask"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.cache[key]
            logger.debug(f"[Cache] EVICTED key={key} (memory freed)")

    def clear(self):
        """
        Remove all entries from the cache and reset cache statistics.

        Each cached entry is evicted individually so that its associated
        data and mask references are released. Hit and miss counters are
        reset to zero after the cache is cleared.
        """
        logger.info(f"[Cache] CLEAR | keys={list(self.cache.keys())}")
        for key in list(self.cache.keys()):
            self._evict(key)
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Return the current cache statistics.

        Returns:
            A dictionary containing the current cache size, configured
            capacity, hit/miss counts, hit rate, and active cache keys.
            ``hit_rate`` is ``0`` when no cache lookups have been performed.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        stats = {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "active_keys": list(self.cache.keys()),
        }
        logger.debug(f"[Cache] STATS: {stats}")
        return stats

    def __contains__(self, key: int) -> bool:
        """
        Check whether a key is currently present in the cache.

        Args:
            key: Integer cache key to check.

        Returns:
            ``True`` if the key exists in the cache, otherwise ``False``.
        """
        return key in self.cache

    def __len__(self) -> int:
        """
        Return the number of entries currently stored in the cache.

        Returns:
            Current number of cached entries.
        """
        return len(self.cache)
