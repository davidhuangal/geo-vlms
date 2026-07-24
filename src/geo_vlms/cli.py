import argparse

from geo_vlms.vlm import build_model_and_processor, prompt_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="The text prompt to send to the VLM.",
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Paths to images to show to the VLM.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        help="The name of the HuggingFace model to use.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="The desired device. E.g., 'cpu', 'mps', 'cuda', etc.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model, processor = build_model_and_processor(args.model_name, args.device)

    result = prompt_model(
        prompt=args.prompt, image_paths=args.images, model=model, processor=processor
    )

    print(result)


if __name__ == "__main__":
    main()
