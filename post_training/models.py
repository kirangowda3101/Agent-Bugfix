"""Model loading shared by training and evaluation, so every model is loaded the same way."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import MODEL, MODEL_REVISION


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else None


def load_tokenizer(padding_side="left"):
    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, padding_side=padding_side)


def load_model(dtype="float32", weights=None, adapter=None):
    """Base model at the pinned revision, or local `weights` (a merged SFT model),
    optionally with a LoRA `adapter` merged on top."""
    if weights:
        model = AutoModelForCausalLM.from_pretrained(weights, dtype=getattr(torch, dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=getattr(torch, dtype))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model
