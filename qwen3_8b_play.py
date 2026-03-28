#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from qwen3_8b_model import Qwen3MiniConfig, Qwen3MiniForCausalLM

DEFAULT_MODEL_PATH = "/ssdshare/yaoy-24/hf_cache/hub/models--Qwen--Qwen3-8B"


def normalize_eos_token_ids(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(x) for x in eos_token_id}


@torch.inference_mode()
def greedy_decode(
    model: Qwen3MiniForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> torch.Tensor:
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids, return_dict=True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)

        if eos_token_ids and int(next_token_id.item()) in eos_token_ids:
            break

    return generated_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run Qwen3-8B manual greedy decoding.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
        help='Transformers device_map, e.g. "cuda:0" or "auto".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else input("Prompt: ").strip()
    if not prompt:
        raise ValueError("Prompt is empty.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script (requested bf16 GPU loading).")
    dtype = torch.bfloat16

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    config = Qwen3MiniConfig.from_pretrained(args.model_path)
    model = Qwen3MiniForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()
    input_device = model.model.embed_tokens.weight.device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(input_device)

    eos_token_ids = normalize_eos_token_ids(model.config.eos_token_id)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(int(tokenizer.eos_token_id))

    full_ids = greedy_decode(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_token_ids,
    )

    prompt_len = input_ids.shape[-1]
    new_ids = full_ids[:, prompt_len:]
    new_text = tokenizer.decode(new_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Decoded Continuation (greedy, manual loop) ===")
    print(new_text)
    print("\n=== Full Text ===")
    print(full_text)


if __name__ == "__main__":
    main()
