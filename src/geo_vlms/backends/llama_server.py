import base64
import mimetypes
import time

import httpx
import openai
from openai import OpenAI, OpenAIError

from .base import Generation, TokenLogprob


def _guess_mime_from_bytes(image_bytes: bytes) -> str:
    # Read the header bytes
    header = image_bytes[:12]

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"

    # Default fallback
    return "image/jpeg"


def _data_url(image: str | bytes) -> str:
    """Convert an image to a data_url ready for llama server."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
        mime_type, _ = mimetypes.guess_type(image)
    elif isinstance(image, bytes):
        image_bytes = image
        mime_type = _guess_mime_from_bytes(image_bytes=image)
    else:
        raise ValueError("Image must be an image path or image data.")

    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    if not mime_type:
        mime_type = "image/jpeg"  # fallback
    return f"data:{mime_type};base64,{base64_string}"


class LlamaServerBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        temperature: float = 0.0,
        seed: int = 0,
        top_k: int = 1,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.top_k = top_k

        # Generous read timeout since the first request can trigger slow
        # prompt processing
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(600.0, connect=5.0)
        )
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=self._http,
        )

        # Simple probe which fails if server isn't reachable
        try:
            self.client.models.list()
        except OpenAIError as e:
            raise ConnectionError(f"Failed to connect to {base_url}.") from e

        # /props lives at the server root
        root = base_url.removesuffix("/v1")
        response = self._http.get(f"{root}/props")
        response.raise_for_status()
        self._props = response.json()

        # A server without a vision projector silently ignores images
        if not self._props.get("modalities", {}).get("vision", False):
            raise RuntimeError(
                f"Server at {base_url} has no vision support; "
                "was it launched with --mmproj?"
            )

    def _build_messages(
        self, prompt: str, images: list[str | bytes] | None = None
    ) -> list:
        """
        Generate the messages format to send to the VLM.

        Args:
            prompt: The text prompt to send to the model.
            images: The paths to images or image bytes to send to the model.

        Returns:
            The messages list in the appropriate format.
        """
        user_content = []

        # Image content
        if images is not None:
            for image in images:
                data_url = _data_url(image=image)

                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )

        # Text content
        user_content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": user_content}]
        return messages

    def generate(
        self,
        prompt: str,
        images: list[str | bytes] | None,
        max_new_tokens: int = 64,
        top_logprobs: int | None = None,
    ) -> Generation:
        """
        Generate a text response from a model.
        """

        chat_kwargs = dict(
            model="unused",
            messages=self._build_messages(prompt=prompt, images=images),
            # Greedy sampling
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=max_new_tokens,
            # top_k and chat_template_kwargs are llama-server-specific.
            extra_body={
                "top_k": self.top_k,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if top_logprobs is not None:
            chat_kwargs["logprobs"] = True
            chat_kwargs["top_logprobs"] = top_logprobs

        t0 = time.perf_counter()
        result = self.client.chat.completions.create(**chat_kwargs)
        latency_s = time.perf_counter() - t0
        text_response = result.choices[0].message.content
        if text_response is None:
            raise RuntimeError("Server response has no message content.")

        if top_logprobs is not None:
            logprobs = result.choices[0].logprobs
            if logprobs is None or logprobs.content is None:
                raise RuntimeError(f"Server at {self.base_url} returned no logprobs.")
            tokens = [
                TokenLogprob(
                    token=tok.token,
                    logprob=tok.logprob,
                    top={t.token: t.logprob for t in tok.top_logprobs},
                )
                for tok in logprobs.content
            ]
        else:
            tokens = None

        usage = result.usage

        generation = Generation(
            text=text_response,
            tokens=tokens,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_s=latency_s,
        )

        return generation

    def describe(self) -> dict:
        meta = {}
        meta["kind"] = "llama_server"
        meta["base_url"] = self.base_url
        meta["openai_version"] = openai.__version__
        meta["sampling"] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "seed": self.seed,
        }
        meta["model_path"] = self._props.get("model_path")
        meta["model_alias"] = self._props.get("model_alias")
        meta["model_ftype"] = self._props.get("model_ftype")
        meta["build_info"] = self._props.get("build_info")
        meta["n_ctx"] = self._props.get("default_generation_settings", {}).get("n_ctx")
        meta["total_slots"] = self._props.get("total_slots")
        return meta
