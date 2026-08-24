import base64
import mimetypes

import httpx
import openai
from openai import OpenAI, OpenAIError


class LlamaServerBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        temperature: float = 0.0,
        seed: int = 0,
        top_k: int = 1,
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.top_k = top_k

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

        # Simple probe which fails if server isn't reachable
        try:
            self.client.models.list()
        except OpenAIError as e:
            raise ConnectionError(f"Failed to connect to {base_url}.") from e

        # /props lives at the server root
        root = base_url.removesuffix("/v1")
        response = httpx.get(f"{root}/props")
        response.raise_for_status()
        self._props = response.json()

        # A server without a vision projector silently ignores images
        if not self._props.get("modalities", {}).get("vision", False):
            raise RuntimeError(
                f"Server at {base_url} has no vision support; "
                "was it launched with --mmproj?"
            )

    def _build_messages(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> list:
        """
        Generate the messages format to send to the VLM.

        Args:
            prompt: The text prompt to send to the model.
            image_paths: The paths to the images to show to the VLM.

        Returns:
            The messages list in the appropriate format.
        """
        user_content = []

        # Image content
        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    # Convert to base64 string
                    image_bytes = image_file.read()
                    base64_string = base64.b64encode(image_bytes).decode("utf-8")

                    # Guess the correct MIME type (e.g., 'image/jpeg', 'image/png')
                    mime_type, _ = mimetypes.guess_type(image_path)
                    if not mime_type:
                        mime_type = "image/jpeg"  # Fallback default

                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_string}"
                            },
                        }
                    )

        # Text content
        user_content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": user_content}]
        return messages

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate a text response from a model.
        """
        result = self.client.chat.completions.create(
            # llama server ignores, but needed for this call
            model="unused",
            messages=self._build_messages(prompt=prompt, image_paths=image_paths),
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
        response = result.choices[0].message.content
        if response is None:
            raise RuntimeError("Server response has no message content.")

        return response

    def describe(self) -> dict:
        meta = {}
        meta["kind"] = "llama-server"
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
