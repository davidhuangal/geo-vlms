# 2. llama-server

Serve a quantized SmolVLM2 with llama.cpp, rerun tutorial 1's questions against it, and compare the results with the HF run.
You need `llama-server` on your PATH ([install](https://github.com/ggml-org/llama.cpp#quick-start)).

## Start the server

```bash
llama-server \
  --hf-repo ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --port 8080
```

`--hf-repo` downloads the Q4_K_M weights and the vision projector (mmproj).
Wait for `listening on http://127.0.0.1:8080`.

## Run

Same questions as tutorial 1, different backend:

```bash
uv run geo-vlms \
  backend=llama_server \
  backend.base_url=http://localhost:8080/v1 \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  task=counting \
  dataset.num_pos=5 \
  dataset.num_neg=2
```

```text
Built 70 vhr10 counting examples. Using HuggingFaceTB/SmolVLM2-2.2B-Instruct via llama_server.
Warning: model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct does not match the server's model ggml-org/SmolVLM2-2.2B-Instruct-GGUF; records will be labeled HuggingFaceTB/SmolVLM2-2.2B-Instruct.
```

The server decides which model runs.
`model_name` only labels records and picks the output path.
Here it's kept the same as tutorial 1 so both runs sit under one model directory:

```text
results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct/
  huggingface/records_seed0.jsonl
  llama_server/records_seed0.jsonl
```

The warning is expected.
The GGUF path, quantization, and llama.cpp build are recorded in `.meta.json` under `backend`.

## Compare

```bash
D=results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct
for backend in huggingface llama_server; do
  uv run scripts/analyze.py \
    --task counting \
    --records $D/$backend/records_seed0.jsonl \
    --metrics valid exact_match absolute_error within_1
done
```

```text
      valid  exact_match  absolute_error  within_1
mean    1.0     0.671429        0.714286  0.842857     # huggingface, bf16
mean    1.0     0.585714        0.657143  0.885714     # llama_server, Q4_K_M
```

Per-question agreement:

```python
from geo_vlms.analysis import load_records
from geo_vlms.tasks import Counting

d = "results/vhr10/counting/HuggingFaceTB/SmolVLM2-2.2B-Instruct"
hf = load_records(f"{d}/huggingface/records_seed0.jsonl").set_index("id")
ll = load_records(f"{d}/llama_server/records_seed0.jsonl").set_index("id")
parse = Counting().parse_response
print(f"{(hf.output.map(parse) == ll.output.map(parse)).mean():.0%} of answers agree")
```

```text
80% of answers agree
```

The two runs don't match, mostly because the model sees a different image.
Compare `prompt_tokens` in the first record of each file: 1106 on HF, 109 on llama-server.
transformers splits each image into tiles, and this GGUF's llama.cpp setup doesn't.
Treat different backends as different runs ([backends.md](../backends.md#comparing-backends)).

## Logprobs

llama-server can return per-token log probabilities:

```bash
uv run geo-vlms \
  backend=llama_server \
  backend.base_url=http://localhost:8080/v1 \
  model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  task=existence \
  dataset.num_pos=2 \
  dataset.no_neg=true \
  top_logprobs=3 \
  out=logprobs.jsonl
```

Each record gains a `tokens` list:

```json
{"token": " No", "logprob": -0.061, "top": {" No": -0.061, " Yes": -2.933, " None": -6.231}}
```

The huggingface backend doesn't support this.

Next: [3. Compare models](03-compare-models.md).
