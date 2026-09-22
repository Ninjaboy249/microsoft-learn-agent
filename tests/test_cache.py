from services.cache import TTLCache


def test_cache_clear_removes_all_items():
    cache = TTLCache(max_size=2, ttl_seconds=60)
    cache.set("one", {"value": 1})
    cache.set("two", {"value": 2})

    cache.clear()

    assert cache.get("one") is None
    assert cache.get("two") is None