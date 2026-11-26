import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # suppress tokenizer parallelism warning

LANG_EMB_OBS_KEY = "lang_emb"

_TOKENIZER_NAME = "openai/clip-vit-large-patch14"
_LANG_MODEL = None
_TOKENIZER = None
_TRANSFORMERS_ERROR = None


def _load_transformers():
    global _TRANSFORMERS_ERROR
    if _TRANSFORMERS_ERROR is not None:
        raise _TRANSFORMERS_ERROR
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except Exception as exc:  # pragma: no cover - depends on optional deps
        _TRANSFORMERS_ERROR = exc
        raise
    return AutoTokenizer, CLIPTextModelWithProjection


def _ensure_lang_model():
    global _LANG_MODEL, _TOKENIZER
    if _LANG_MODEL is not None and _TOKENIZER is not None:
        return _LANG_MODEL, _TOKENIZER

    AutoTokenizer, CLIPTextModelWithProjection = _load_transformers()
    cache_dir = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
    _LANG_MODEL = CLIPTextModelWithProjection.from_pretrained(
        _TOKENIZER_NAME,
        cache_dir=cache_dir,
    ).eval()
    _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_NAME, TOKENIZERS_PARALLELISM=True)
    return _LANG_MODEL, _TOKENIZER


def get_lang_emb(lang):
    if lang is None:
        return None

    lang_emb_model, tokenizer = _ensure_lang_model()
    tokens = tokenizer(
        text=lang,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    lang_emb = lang_emb_model(**tokens)["text_embeds"].detach()[0]
    return lang_emb


def get_lang_emb_shape():
    return list(get_lang_emb("dummy").shape)
