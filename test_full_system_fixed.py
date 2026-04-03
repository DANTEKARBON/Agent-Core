#!/usr/bin/env python3
import sys
import os
import time
import tempfile
import pytest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cache import TTLCache
from core.model_registry import ModelRegistry
from core.contracts import LLMResponse
from core.resilience import with_retry, with_timeout, CircuitBreaker
from adapters.factory import LLMAdapterFactory
from adapters.mlx_openai_adapter import MLXOpenAIAdapter

# ========== Тесты кэша ==========
def test_cache_basic():
    cache = TTLCache(ttl_seconds=1, max_size=10)
    cache.set("key", "value")
    assert cache.get("key") == "value"
    time.sleep(1.1)
    assert cache.get("key") is None

def test_cache_max_size():
    cache = TTLCache(ttl_seconds=10, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_cache_save_load():
    cache = TTLCache(ttl_seconds=10)
    cache.set("x", 100)
    cache.set("y", 200)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        tmpfile = f.name
    cache.save(tmpfile)
    new_cache = TTLCache()
    new_cache.load(tmpfile)
    assert new_cache.get("x") == 100
    assert new_cache.get("y") == 200
    os.unlink(tmpfile)

# ========== Тесты ModelRegistry (сброс синглтона) ==========
@pytest.fixture
def temp_config():
    config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    config.write("""
models:
  phi3:
    path: "/fake/path/phi3"
    port: 10240
  coder:
    path: "/fake/path/coder"
    port: 10241
""")
    config.close()
    yield config.name
    os.unlink(config.name)

def test_registry_load(temp_config):
    # Сбрасываем синглтон, чтобы загрузить временный конфиг
    ModelRegistry._instance = None
    registry = ModelRegistry(config_path=temp_config)
    assert set(registry.list_models()) == {"phi3", "coder"}
    info = registry.get_model_info("phi3")
    assert info["port"] == 10240
    assert info["path"] == "/fake/path/phi3"

def test_registry_invalid_model(temp_config):
    ModelRegistry._instance = None
    registry = ModelRegistry(config_path=temp_config)
    with pytest.raises(KeyError):
        registry.get_model_info("unknown")

# ========== Тесты CircuitBreaker ==========
def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    assert cb.state == "closed"
    
    def success():
        return "ok"
    assert cb.call(success) == "ok"
    assert cb.failures == 0
    
    def fail():
        raise ValueError("fail")
    with pytest.raises(ValueError):
        cb.call(fail)
    assert cb.failures == 1
    with pytest.raises(ValueError):
        cb.call(fail)
    assert cb.failures == 2
    assert cb.state == "open"
    
    with pytest.raises(Exception) as exc:
        cb.call(success)
    assert "Circuit breaker is open" in str(exc.value)
    
    time.sleep(1.1)
    assert cb.call(success) == "ok"
    assert cb.state == "closed"

# ========== Тесты декораторов ==========
def test_retry():
    counter = 0
    @with_retry(attempts=3, delay=0.1)
    def flaky():
        nonlocal counter
        counter += 1
        if counter < 2:
            raise ValueError("fail")
        return "success"
    assert flaky() == "success"
    assert counter == 2

def test_timeout():
    @with_timeout(0.5)
    def slow():
        time.sleep(1)
        return "done"
    with pytest.raises(Exception) as exc:
        slow()
    assert "timed out" in str(exc.value)

# ========== Тесты адаптеров ==========
def test_mock_adapter():
    config = {"type": "mock", "response": "test response"}
    adapter = LLMAdapterFactory.create(config)
    result = adapter.generate("hello")
    assert result["response"] == "test response"
    # MockAdapter возвращает metadata без поля status, проверяем только наличие response
    assert "response" in result
    assert "metadata" in result

@patch('adapters.mlx_openai_adapter.requests.Session')
def test_mlx_adapter_generate(mock_session_class):
    mock_session = Mock()
    mock_session_class.return_value = mock_session
    mock_response = Mock()
    mock_response.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        b'data: {"choices":[{"delta":{"content":" world"}}]}',
        b'data: [DONE]'
    ]
    mock_response.status_code = 200
    mock_session.post.return_value = mock_response
    
    adapter = MLXOpenAIAdapter(base_url="http://fake", model_name="test", model_manager=None)
    result = adapter.generate("hi")
    assert "Hello world" in result["response"]
    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["ttft_ms"] is not None

# ========== Тесты LLMResponse ==========
def test_llm_response_from_dict():
    data = {
        "text": "response text",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "status": "success",
        "ttft_ms": 123.4
    }
    resp = LLMResponse.from_dict(data)
    assert resp.text == "response text"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 20
    assert resp.ttft_ms == 123.4

# ========== Тест graceful shutdown кэша ==========
def test_cache_graceful_shutdown():
    cache = TTLCache(ttl_seconds=10)
    cache.set("persist", "data")
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        tmpfile = f.name
    cache.save(tmpfile)
    new_cache = TTLCache()
    new_cache.load(tmpfile)
    assert new_cache.get("persist") == "data"
    os.unlink(tmpfile)

# ========== Тест estimate_tokens ==========
def test_estimate_tokens():
    adapter = MLXOpenAIAdapter(base_url="http://fake", model_name="test")
    assert adapter._estimate_tokens("hello") == 1
    assert adapter._estimate_tokens("hello world") == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
