import json
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
    from peft import PeftModel
except ImportError:
    AutoProcessor = None
    AutoModelForImageTextToText = None
    pipeline = None
    PeftModel = None

MODEL_ID = "DestinBir/buzy-ai-gemma4-lora"
WHISPER_MODEL_ID = "openai/whisper-small"
MAX_NEW_TOKENS = 1024
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
        path = hf_hub_download(repo_id=MODEL_ID, filename="adapter_config.json")
        with open(path) as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", MODEL_ID)
    except Exception:
        return "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"


def load_models():
    if _bundle.loaded:
        return _bundle

    if AutoProcessor is None or AutoModelForImageTextToText is None:
        raise RuntimeError(
            "transformers/torch not installed. Run: "
            "pip install torch transformers accelerate peft"
        )

    print(f"[Buzy AI] Loading multimodal model '{MODEL_ID}' on {DEVICE} ...")
    _bundle.processor = AutoProcessor.from_pretrained(MODEL_ID)

    try:
        _bundle.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
    except Exception:
        base_model_id = _get_base_model_id()
        print(f"[Buzy AI] Loading base model '{base_model_id}' and applying LoRA adapter ...")
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        _bundle.model = PeftModel.from_pretrained(base, MODEL_ID)

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
