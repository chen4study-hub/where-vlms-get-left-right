"""Loading the models and scoring multiple-choice answers from next-token logits.

Every experiment scores a question by one forward pass and compares the logits of the option letters at the
last position; nothing is sampled unless an experiment says so. Models are loaded with sdpa attention on one
GPU; bf16 unless an experiment asks for fp32. torch is imported on use, so that the CPU analyses, which import
constants from the experiment modules, run without it.
"""

QWEN_MAX_PIXELS = 1280 * 28 * 28           # processor max_pixels used throughout (Qwen2.5-VL default)


def load_qwen(name, dtype="bfloat16", max_pixels=QWEN_MAX_PIXELS, min_pixels=None, device_map="cuda", **proc_kw):
    """Qwen2.5-VL model (eval mode) and its processor."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype, device_map=device_map, attn_implementation="sdpa").eval()
    if min_pixels is not None:
        proc_kw["min_pixels"] = min_pixels
    proc = AutoProcessor.from_pretrained(name, max_pixels=max_pixels, **proc_kw)
    return model, proc


def load_internvl(name, dtype="bfloat16", device_map="cuda"):
    """InternVL3 (-hf, transformers-native) model (eval mode) and its processor."""
    import torch
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    model = InternVLForConditionalGeneration.from_pretrained(
        name, torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype, device_map=device_map, attn_implementation="sdpa").eval()
    return model, AutoProcessor.from_pretrained(name)


def load(name, family, **kw):
    return (load_qwen if family == "qwen" else load_internvl)(name, **kw)


def chat_text(proc, text, images=1):
    """Chat-formatted prompt with `images` image placeholders before the text."""
    content = [{"type": "image"} for _ in range(images)] + [{"type": "text", "text": text}]
    return proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)


def build_inputs(proc, image, text, family="qwen", device="cuda"):
    """Processor inputs for one image and one question. InternVL uses a single 448 x 448 tile."""
    extra = {"crop_to_patches": False} if family == "internvl" else {}
    images = [image] if image is not None else None
    chat = chat_text(proc, text, images=1 if image is not None else 0)
    return proc(text=[chat], images=images, return_tensors="pt", **extra).to(device)


def letter_ids(tokenizer, letters="AB"):
    """Token ids of the option letters; they must be distinct single tokens."""
    ids = [tokenizer.encode(L, add_special_tokens=False)[0] for L in letters]
    assert len(set(ids)) == len(ids), f"option letters share token ids: {ids}"
    return ids


def option_logprobs(logits, ids):
    """log-softmax over the option letters only (list of floats)."""
    import torch
    return torch.log_softmax(logits[ids], -1).tolist()


class FP32OptionScorer:
    """fp32 logits of the option tokens at the last position, from the hidden state entering lm_head.

    In bf16 the two option logits are occasionally exactly equal, and a tie would always go to the first option; the
    experiments that score in fp32 multiply the lm_head input by the option rows in fp32 instead (remaining exact
    ties are recorded as ties). Call scores() (or the object) after a forward pass of `model`.
    """

    def __init__(self, model, token_ids):
        self.last = {}
        model.lm_head.register_forward_pre_hook(lambda m, args: self.last.__setitem__("h", args[0][0, -1].detach()))
        self.weight = model.lm_head.weight[token_ids].float()

    def scores(self):
        return self.last["h"].float() @ self.weight.t()

    __call__ = scores


def greedy_text(model, tokenizer, inputs, max_new_tokens):
    """Greedy continuation of one prompt, decoded without special tokens (not stripped)."""
    model.model.rope_deltas = None
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def pick(choices, lp):
    """Two-option prediction from option log-probs; an exact tie goes to the first option."""
    return choices[int(lp[1] > lp[0])]
