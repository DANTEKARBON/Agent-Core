import hashlib
import time
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TTLCache:
    def __init__(self, ttl_seconds: int = 3600, enabled: bool = True):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, model: str, params: Dict) -> str:
        key_data = f"{prompt}:{model}:{sorted(params.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, prompt: str, model: str, params: Optional[Dict] = None) -> Optional[Any]:
        if not self.enabled:
            return None
        if params is None:
            params = {}
        key = self._make_key(prompt, model, params)
        entry = self._cache.get(key)
        if entry and entry['expires'] > time.time():
            self.hits += 1
            logger.debug(f"Cache HIT: key={key[:8]}...")
            return entry['value']
        elif entry:
            del self._cache[key]
        self.misses += 1
        logger.debug(f"Cache MISS: key={key[:8]}...")
        return None

    def set(self, prompt: str, model: str, value: Any, params: Optional[Dict] = None):
        if not self.enabled:
            return
        if params is None:
            params = {}
        key = self._make_key(prompt, model, params)
        self._cache[key] = {
            'value': value,
            'expires': time.time() + self.ttl
        }
        logger.debug(f"Cache SET: key={key[:8]}..., ttl={self.ttl}")

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }
