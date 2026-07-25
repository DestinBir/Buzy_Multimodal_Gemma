import json
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
        pipeline,
    )
    from peft import PeftModel
except ImportError:
    AutoProcessor = None
    AutoModelForImageTextToText = None
    BitsAndBytesConfig = None
    pipeline = None
    PeftModel = None

LORA_ADAPTER_ID = "DestinBir/buzy-ai-gemma4-lora"
WHISPER_MODEL_ID = "openai/whisper-small"
MAX_NEW_TOKENS = 1024
MAX_PROMPT_TOKENS = 4096
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


@dataclass
class ModelBundle:
    processor: Optional[object] = None
    model: Optional[object] = None
    asr_pipe: Optional[object] = None
    loaded: bool = False


_bundle = ModelBundle()


def _get_base_model_id() -> str:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=LORA_ADAPTER_ID, filename="adapter_config.json")
        with open(path) as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", LORA_ADAPTER_ID)
    except Exception:
        return "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"


def load_models():
    if _bundle.loaded:
        return _bundle

    if AutoProcessor is None or AutoModelForImageTextToText is None:
        raise RuntimeError(
            "transformers/torch not installed. Run: "
            "pip install torch transformers accelerate peft bitsandbytes"
        )

    base_model_id = _get_base_model_id()
    print(f"[Buzy AI] Loading base model '{base_model_id}' (4-bit) on {DEVICE} ...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    _bundle.processor = AutoProcessor.from_pretrained(LORA_ADAPTER_ID)

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print(f"[Buzy AI] Applying LoRA adapter '{LORA_ADAPTER_ID}' ...")
    _bundle.model = PeftModel.from_pretrained(base, LORA_ADAPTER_ID)

    if DEVICE != "cuda":
        _bundle.model.to(DEVICE)

    print(f"[Buzy AI] Loading Whisper ASR model '{WHISPER_MODEL_ID}' ...")
    _bundle.asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=0 if DEVICE == "cuda" else -1,
    )

    _bundle.loaded = True
    print("[Buzy AI] Models loaded.")
    return _bundle


def count_tokens(text: str) -> int:
    bundle = load_models()
    tokens = bundle.processor.tokenizer.encode(text)
    return len(tokens)


def run_gemma(messages: list) -> str:
    bundle = load_models()
    processor = bundle.processor
    model = bundle.model

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    if prompt_len > MAX_PROMPT_TOKENS:
        raise RuntimeError(
            f"Prompt too long ({prompt_len} tokens > {MAX_PROMPT_TOKENS} max). "
            "Try uploading fewer or shorter documents."
        )

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, input_len:]
    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return output_text.strip()
