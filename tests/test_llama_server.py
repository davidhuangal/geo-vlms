import json

import httpx
import pytest
from openai import APIStatusError

from geo_vlms.backends.llama_server import LlamaServerBackend

BASE_URL = "http://server/v1"

PROPS = {
    "model_path": "/models/fake-Q4_K_M.gguf",
    "model_alias": "org/fake:Q4_K_M",
    "model_ftype": "Q4_K - Medium",
    "build_info": "b1234-abc",
    "total_slots": 4,
    "default_generation_settings": {"n_ctx": 16384},
    "modalities": {"vision": True, "video": False, "audio": False},
}

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "fake",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "canned response"},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}

LOGPROBS_RESPONSE = json.loads(json.dumps(CHAT_RESPONSE))
LOGPROBS_RESPONSE["choices"][0]["logprobs"] = {
    "content": [
        {
            "token": "Yes",
            "logprob": -0.1,
            "bytes": None,
            "top_logprobs": [
                {"token": "Yes", "logprob": -0.1, "bytes": None},
                {"token": "No", "logprob": -2.3, "bytes": None},
            ],
        }
    ]
}


def build_backend(requests, props=PROPS, chat_response=CHAT_RESPONSE, chat_status=200):
    """Backend wired to a fake server; captures every request in `requests`."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/props":
            return httpx.Response(200, json=props)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(chat_status, json=chat_response)
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return LlamaServerBackend(base_url=BASE_URL, http_client=http_client)


def test_generate_request_body(tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake image bytes")

    requests = []
    backend = build_backend(requests)
    output = backend.generate("how many?", [str(image_path)], max_new_tokens=32)

    assert output.text == "canned response"
    assert output.tokens is None
    assert output.prompt_tokens == 10
    assert output.completion_tokens == 3

    body = json.loads(requests[-1].read())
    assert "logprobs" not in body
    assert "top_logprobs" not in body

    # Greedy sampling must be requested explicitly: server defaults are not
    # greedy, and hybrid reasoning models must not think away the budget.
    assert body["temperature"] == 0.0
    assert body["seed"] == 0
    assert body["top_k"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 32

    (message,) = body["messages"]
    image_part, text_part = message["content"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert text_part == {"type": "text", "text": "how many?"}


def test_generate_bytes_image():
    requests = []
    backend = build_backend(requests)
    backend.generate("q", [b"\x89PNG\r\n\x1a\nfake"], max_new_tokens=16)

    body = json.loads(requests[-1].read())
    image_part, _ = body["messages"][0]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_logprobs():
    requests = []
    backend = build_backend(requests, chat_response=LOGPROBS_RESPONSE)
    output = backend.generate("q", None, max_new_tokens=16, top_logprobs=2)

    body = json.loads(requests[-1].read())
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2

    (tok,) = output.tokens
    assert tok.token == "Yes"
    assert tok.logprob == -0.1
    assert tok.top == {"Yes": -0.1, "No": -2.3}


def test_generate_logprobs_missing_raises():
    backend = build_backend([])
    with pytest.raises(RuntimeError, match="logprobs"):
        backend.generate("q", None, max_new_tokens=16, top_logprobs=2)


def test_generate_without_images():
    requests = []
    backend = build_backend(requests)
    backend.generate("text only", None, max_new_tokens=16)

    body = json.loads(requests[-1].read())
    (message,) = body["messages"]
    assert message["content"] == [{"type": "text", "text": "text only"}]


def test_generate_null_content_raises():
    null_response = json.loads(json.dumps(CHAT_RESPONSE))
    null_response["choices"][0]["message"]["content"] = None

    backend = build_backend([], chat_response=null_response)
    with pytest.raises(RuntimeError, match="no message content"):
        backend.generate("q", None, max_new_tokens=16)


def test_generate_http_error_raises():
    # Construction succeeds (probe and /props are served); only chat fails
    backend = build_backend([], chat_response={"error": "boom"}, chat_status=500)
    with pytest.raises(APIStatusError):
        backend.generate("q", None, max_new_tokens=16)


def test_unreachable_server_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "loading"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectionError):
        LlamaServerBackend(base_url=BASE_URL, http_client=http_client)


def test_missing_vision_raises():
    no_vision = json.loads(json.dumps(PROPS))
    no_vision["modalities"]["vision"] = False

    with pytest.raises(RuntimeError, match="vision"):
        build_backend([], props=no_vision)


def test_describe():
    backend = build_backend([])
    meta = backend.describe()

    assert meta["kind"] == "llama_server"
    assert meta["base_url"] == BASE_URL
    assert meta["sampling"] == {"temperature": 0.0, "top_k": 1, "seed": 0}
    assert meta["model_path"] == "/models/fake-Q4_K_M.gguf"
    assert meta["model_alias"] == "org/fake:Q4_K_M"
    assert meta["model_ftype"] == "Q4_K - Medium"
    assert meta["build_info"] == "b1234-abc"
    assert meta["n_ctx"] == 16384
    assert meta["total_slots"] == 4
