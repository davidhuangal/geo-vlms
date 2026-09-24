# Backends

Select with `backend=huggingface` or `backend=llama_server` (default).
Both decode greedily with thinking off.

## Install

| Install | Backends |
|---|---|
| `uv sync` | both |
| `pip install .` | `llama_server` |
| `pip install ".[hf]"` | both |

The `hf` extra pulls torch and transformers.
The core install doesn't, so a llama-server-only machine or container stays small.

## huggingface

Loads the model in-process with `AutoModelForImageTextToText`.

```bash
uv run geo-vlms \
  backend=huggingface \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

| Key | Default | Meaning |
|---|---|---|
| `model_name` | | HF repo id. Also labels the records. |
| `backend.device` | auto | `cuda`, `mps`, or `cpu`. Auto picks the first available. |

Fixed choices:
- dtype `bfloat16`, or `float16` on CUDA without bf16 support.
- Attention `sdpa` on CUDA, `eager` elsewhere.
- `do_sample=False`, `enable_thinking=False` in the chat template.

Not supported: `top_logprobs` (raises), `latency_s` (recorded as `null`).

## llama_server

Talks to a [llama-server](https://github.com/ggml-org/llama.cpp) over its OpenAI-compatible API.

```bash
llama-server \
  --hf-repo ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --port 8080
```

```bash
uv run geo-vlms \
  backend=llama_server \
  backend.base_url=http://localhost:8080/v1 \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

| Key | Default | Meaning |
|---|---|---|
| `backend.base_url` | required | Server's `/v1` URL. |
| `model_name` | | Labels the records only. The server decides the model. |
| `backend.temperature` | `0.0` | |
| `backend.top_k` | `1` | |
| `top_logprobs` | `null` | Top alternatives to record per generated token. |

Auth: set `GEO_VLMS_LLAMA_API_KEY`.
It defaults to `unused`.

Startup checks:
- `/v1/models` must answer, or the run fails with `ConnectionError`.
- `/props` must report vision support. Without `--mmproj` the server silently drops images, so this fails fast.
- If `model_name` differs from the server's model alias, it warns and keeps `model_name`.

Requests time out after 600 s and aren't retried.
The sampling seed sent to the server is always 0.
It doesn't follow `seed`, which only affects dataset sampling.

Results are deterministic for a given GGUF, llama.cpp build, and server flags.
`.meta.json` records `model_path`, `model_ftype`, `build_info`, `n_ctx`, and `total_slots` from `/props`.

## Comparing backends

Treat each backend as a separate run.
The same model can answer differently on huggingface and llama_server, so compare models within one backend.

Common causes:
- Image preprocessing differs. A large gap in `prompt_tokens` between the two runs' records means the model saw a different image.
- Quantized weights change answers.
- llama.cpp versions differ. Builds before commit `163a407` mishandle Gemma 4 image attention.

## Container

```bash
docker build --tag geo-vlms .
docker run --rm \
  --volume ./data:/app/data \
  --volume ./results:/app/results \
  geo-vlms \
  geo-vlms \
    backend=llama_server \
    backend.base_url=http://host:8080/v1 \
    model_name=org/model-name
```

The image has the core install only, so it holds no torch or GPU runtime.
Build args for a private base image and package mirror:

| Arg | Default |
|---|---|
| `BASE_IMAGE` | `python:3.13-slim` |
| `PIP_INDEX_URL` | `https://pypi.org/simple` |
| `PIP_EXTRA_INDEX_URL` | empty |

`uv.lock` points at public indexes.
Behind a mirror, install with pip instead of `uv sync --locked`.
