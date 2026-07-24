import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def build_messages(prompt: str, image_paths: list[str] | None = None) -> list:
    """
    Generate the messages format to send to the VLM.

    Args:
        prompt: The text prompt to send to the model.
        image_paths: The paths to the images to show to the VLM.

    Returns:
        The messages list in the appropriate format.
    """
    # Build the content of the message
    user_content = []
    if image_paths is not None:
        for image_path in image_paths:
            user_content.append({"type": "image", "path": str(image_path)})
    user_content.append({"type": "text", "text": prompt})

    # Build the message list
    messages = [{"role": "user", "content": user_content}]

    return messages


def build_model_and_processor(model_name: str, device: str):
    """
    Build the HuggingFace VLM and its associated processor.

    Args:
        model_name: The HuggingFace name for the model.
        device: The desired device for the model. E.g. 'cuda' or 'mps'.

    Returns:
        The instantiated model and its associated processor.
    """
    # Init model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if "cuda" in device else "eager",
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor


def preprocess_inputs(
    prompt: str, model, processor, image_paths: list[str] | None = None
):
    """
    Preprocess user inputs such that they can be consumed by the VLM.

    Args:
        prompt: The user text prompt.
        model: The VLM model.
        processor: The processor associated with the model.
        image_paths: The paths to the images to show to the VLM.

    Returns:
        The inputs in the format consumable by the model, on the same
        device as the model.
    """
    # Convert prompt / images into the expected messages format
    messages = build_messages(prompt=prompt, image_paths=image_paths)

    # Convert the messages into the format the VLM expects
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    return inputs


def prompt_model(prompt, image_paths, model, processor):
    """
    Prompt a VLM with text and images.

    Args:
        prompt: The user text prompt.
        model: The VLM model.
        processor: The processor associated with the model.

    Returns:
        The generated text from the model.
    """

    # Prepare inputs to send to the model
    inputs = preprocess_inputs(
        prompt=prompt, model=model, processor=processor, image_paths=image_paths
    )

    # Generate the raw tokens from the model
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

    # Grab only the tokens which correspond to the model output
    prompt_len = inputs["input_ids"].shape[-1]
    new_ids = generated_ids[:, prompt_len:]

    # Decode text from the raw tokens
    generated_text = processor.batch_decode(new_ids, skip_special_tokens=True)

    return generated_text[0]
