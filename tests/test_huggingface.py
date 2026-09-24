import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from geo_vlms.backends.huggingface import HuggingFaceBackend

MODEL_NAME = "org/fake-model"


@pytest.fixture
def hf_mocks(monkeypatch):
    model = Mock(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        config=SimpleNamespace(_commit_hash="abc123", _attn_implementation="eager"),
        generation_config=SimpleNamespace(pad_token_id=None),
    )
    model.to.return_value = model
    processor = Mock(
        tokenizer=SimpleNamespace(pad_token_id=42),
        chat_template="{% if enable_thinking %}<think>{% endif %}",
    )
    load_model = Mock(return_value=model)
    load_processor = Mock(return_value=processor)
    bf16_supported = Mock(return_value=True)

    monkeypatch.setattr(
        "geo_vlms.backends.huggingface.AutoModelForImageTextToText.from_pretrained",
        load_model,
    )
    monkeypatch.setattr(
        "geo_vlms.backends.huggingface.AutoProcessor.from_pretrained", load_processor
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16_supported)

    return SimpleNamespace(
        model=model,
        processor=processor,
        load_model=load_model,
        load_processor=load_processor,
        bf16_supported=bf16_supported,
    )


@pytest.fixture
def backend(hf_mocks):
    return HuggingFaceBackend(MODEL_NAME, "cpu")


@pytest.mark.parametrize(
    ("device", "bf16_supported", "dtype", "attention"),
    [
        ("cpu", False, torch.bfloat16, "eager"),
        ("mps", False, torch.bfloat16, "eager"),
        ("cuda", True, torch.bfloat16, "sdpa"),
        ("cuda", False, torch.float16, "sdpa"),
        ("cuda:1", True, torch.bfloat16, "sdpa"),
        ("cuda:1", False, torch.float16, "sdpa"),
    ],
)
def test_initialization(hf_mocks, device, bf16_supported, dtype, attention):
    hf_mocks.bf16_supported.return_value = bf16_supported

    backend = HuggingFaceBackend(MODEL_NAME, device)

    hf_mocks.load_model.assert_called_once_with(
        MODEL_NAME, dtype=dtype, attn_implementation=attention
    )
    hf_mocks.model.to.assert_called_once_with(device)
    hf_mocks.load_processor.assert_called_once_with(MODEL_NAME)
    assert backend.model_name == MODEL_NAME
    assert backend.model is hf_mocks.model
    assert backend.processor is hf_mocks.processor
    assert backend.model.generation_config.pad_token_id == 42
    if "cuda" in device:
        hf_mocks.bf16_supported.assert_called_once_with()
    else:
        hf_mocks.bf16_supported.assert_not_called()


@pytest.mark.parametrize(
    ("cuda", "mps", "device"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_device_autodetect(hf_mocks, monkeypatch, cuda, mps, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    HuggingFaceBackend(MODEL_NAME, device=None)

    hf_mocks.model.to.assert_called_once_with(device)


@pytest.mark.parametrize("images", [None, []])
def test_messages_without_images(backend, images):
    assert backend._build_messages("text only", images) == [
        {"role": "user", "content": [{"type": "text", "text": "text only"}]}
    ]


def test_messages_with_images(backend):
    assert backend._build_messages("compare", ["a.jpg", Path("b.jpg")]) == [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "a.jpg"},
                {"type": "image", "path": "b.jpg"},
                {"type": "text", "text": "compare"},
            ],
        }
    ]


def test_messages_with_image_bytes(backend):
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")

    (message,) = backend._build_messages("q", [buf.getvalue()])
    image_part, text_part = message["content"]

    assert image_part["type"] == "image"
    assert isinstance(image_part["image"], Image.Image)
    assert image_part["image"].size == (2, 2)
    assert text_part == {"type": "text", "text": "q"}


def test_generate_logprobs_raises(backend):
    with pytest.raises(NotImplementedError):
        backend.generate("q", None, top_logprobs=2)


@pytest.mark.parametrize(
    ("images", "token_options", "new_tokens", "expected_text"),
    [
        (["a.jpg"], {"max_new_tokens": 16}, [20, 21], "two planes"),
        (None, {}, [20], "hello"),
        ([], {}, [], ""),
    ],
)
def test_generate(backend, hf_mocks, images, token_options, new_tokens, expected_text):
    input_ids = torch.tensor([[10, 11, 12]])
    attention_mask = torch.ones_like(input_ids)
    inputs = BatchFeature({"input_ids": input_ids, "attention_mask": attention_mask})
    if images:
        inputs["pixel_values"] = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    hf_mocks.processor.apply_chat_template.return_value = inputs
    hf_mocks.model.generate.return_value = torch.tensor([[10, 11, 12, *new_tokens]])
    hf_mocks.processor.batch_decode.return_value = [expected_text]

    output = backend.generate("q", images, **token_options)
    assert output.text == expected_text
    assert output.tokens is None
    assert output.prompt_tokens == 3
    assert output.completion_tokens == len(new_tokens)

    content = [{"type": "image", "path": path} for path in images or []]
    content.append({"type": "text", "text": "q"})
    hf_mocks.processor.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    hf_mocks.model.generate.assert_called_once()
    args, kwargs = hf_mocks.model.generate.call_args
    assert args == ()
    assert set(kwargs) == {*inputs, "do_sample", "max_new_tokens"}
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == token_options.get("max_new_tokens", 64)
    torch.testing.assert_close(kwargs["input_ids"], input_ids)
    torch.testing.assert_close(kwargs["attention_mask"], attention_mask)
    if images:
        # Real BatchFeature conversion must cast pixels while keeping IDs integral.
        torch.testing.assert_close(
            kwargs["pixel_values"],
            torch.ones((1, 3, 2, 2), dtype=backend.model.dtype),
        )

    hf_mocks.processor.batch_decode.assert_called_once()
    args, kwargs = hf_mocks.processor.batch_decode.call_args
    (decoded_ids,) = args
    torch.testing.assert_close(
        decoded_ids, torch.tensor([new_tokens], dtype=torch.long)
    )
    assert kwargs == {"skip_special_tokens": True}


def test_generate_omits_unused_thinking_switch(hf_mocks):
    hf_mocks.processor.chat_template = "{{ messages }}"
    backend = HuggingFaceBackend(MODEL_NAME, "cpu")
    inputs = BatchFeature({"input_ids": torch.tensor([[10]])})
    hf_mocks.processor.apply_chat_template.return_value = inputs
    hf_mocks.model.generate.return_value = torch.tensor([[10, 20]])
    hf_mocks.processor.batch_decode.return_value = ["hi"]

    backend.generate("q", None)

    _, kwargs = hf_mocks.processor.apply_chat_template.call_args
    assert "enable_thinking" not in kwargs


def test_describe(backend):
    assert backend.describe() == {
        "kind": "huggingface",
        "name": MODEL_NAME,
        "commit_hash": "abc123",
        "attn_implementation": "eager",
        "dtype": "bfloat16",
        "device": "cpu",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
