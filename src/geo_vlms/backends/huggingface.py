import io

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import Generation


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HuggingFaceBackend:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        """
        HuggingFace-powered backend.

        Args:
            model_name: The `organization/model-name` for the desired model.
            device: The desired PyTorch device, e.g. 'cuda', 'mps', 'cpu'.
                None picks the first available of cuda, mps, cpu.
        """
        self.model_name = model_name
        if device is None:
            device = _default_device()

        # Pick dtype based on hardware support
        if "cuda" in device and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        # Model creation
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=dtype,
            attn_implementation="sdpa" if "cuda" in device else "eager",
        ).to(device)

        # Associated processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.pad_token_id
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
        # Build the content of the message
        user_content = []
        if images is not None:
            for image in images:
                if isinstance(image, bytes):
                    pil_image = Image.open(io.BytesIO(image))
                    user_content.append({"type": "image", "image": pil_image})
                else:
                    user_content.append({"type": "image", "path": str(image)})
        user_content.append({"type": "text", "text": prompt})

        # Build the message list
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
        Prompt a VLM with text and images.

        Args:
            prompt: The user text prompt.
            images: The paths to images or image bytes to send to the model.
            max_new_tokens: Sets the max tokens a model is allowed to output.
            top_logprobs: Unsupported by this backend; must be None.

        Returns:
            The generated text and token counts.
        """
        if top_logprobs is not None:
            raise NotImplementedError("HuggingFaceBackend does not return logprobs.")

        # Convert prompt / images into the expected messages format
        messages = self._build_messages(prompt=prompt, images=images)

        # Convert the messages into the tensors the VLM expects, on the model's
        # device. Only floating-point tensors are cast, so input_ids stays integral.
        # return_tensors must be a top-level kwarg: inside processor_kwargs it is
        # ignored for text tokenization by every processor here except Qwen3.5's,
        # leaving input_ids as plain lists that crash model.generate.
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device, dtype=self.model.dtype)

        # Generate the raw tokens from the model
        generated_ids = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )

        # Grab only the tokens which correspond to the model output
        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = generated_ids[:, prompt_len:]

        # Decode text from the raw tokens
        generated_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)

        return Generation(
            text=generated_text[0],
            prompt_tokens=prompt_len,
            completion_tokens=new_ids.shape[-1],
        )

    def describe(self) -> dict:
        meta = {}
        meta["kind"] = "huggingface"
        meta["name"] = self.model_name
        meta["commit_hash"] = self.model.config._commit_hash
        meta["attn_implementation"] = self.model.config._attn_implementation
        meta["dtype"] = str(self.model.dtype).removeprefix("torch.")
        meta["device"] = str(self.model.device)
        meta["transformers_version"] = transformers.__version__
        meta["torch_version"] = torch.__version__
        return meta
