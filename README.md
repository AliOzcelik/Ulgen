# Ülgen

Turkish specific, from scratch LLM with its own trained tokenizer; with the goal of practicing and learning.

## Tokenizers

Turkish **SentencePiece Unigram** tokenizers trained on the
[vngrs-web-corpus](https://huggingface.co/datasets/vngrs-ai/vngrs-web-corpus)
(cleaned Turkish OSCAR + mC4), exported to HuggingFace format. Shared config:
`byte_fallback` (never out-of-vocabulary), `split_digits` (numbers split per digit),
whitespace preserved, NFKC normalization **without** case folding (Turkish İ/ı/I/i
and casing kept). Special tokens: `<unk>=0 <s>=1 </s>=2 <pad>=3`.

| Folder | Vocab | Fertility (tok/word)¹ |
|---|--:|--:|
| [`tokenizers/turkish-unigram-32k`](tokenizers/turkish-unigram-32k) | 32,000 | 1.564 |
| [`tokenizers/turkish-unigram-48k`](tokenizers/turkish-unigram-48k) | 48,000 | 1.501 |
| [`tokenizers/turkish-unigram-64k`](tokenizers/turkish-unigram-64k) | 64,000 | 1.461 |

¹ Tokens per word on a held-out Turkish sample — lower is better. For reference on the
same text: BERTurk 1.55, XLM-R 1.82, GPT-4o 2.19, GPT-2 3.77 — i.e. these beat every
general LLM tokenizer on Turkish. Bigger vocab lowers fertility with diminishing
returns (32k→64k: −6.6% tokens for double the embedding size), so **32k** is the
efficient default and **48k/64k** trade parameters for shorter sequences.

Each folder contains `tokenizer.json` + configs (HF fast tokenizer) and the original
SentencePiece `.model`.

### Use in Python

```python
# HuggingFace fast tokenizer (no `transformers` needed)
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizers/turkish-unigram-32k/tokenizer.json")
enc = tok.encode("Türkiye'nin başkenti Ankara'dır.")
print(enc.tokens, enc.ids)

# or with transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizers/turkish-unigram-32k")

# or raw SentencePiece
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizers/turkish-unigram-32k/tr_unigram_32k.model")
```

## Dashboard

[`dashboard/compare_dashboard.py`](dashboard/compare_dashboard.py) is a small Flask app
that tokenizes a sentence you type with **every tokenizer at once** — the three Turkish
ones above plus open LLM tokenizers (BERTurk, XLM-R, Qwen, Llama 3, Gemma 3, GPT-2,
GPT-4o/o200k, …) — and shows a **bar chart of token counts** and the **colored token
stream** per model. It has a light/dark toggle and a PNG export for reports.

```bash
cd dashboard
pip install flask tokenizers sentencepiece huggingface_hub tiktoken
python3 compare_dashboard.py          # then open http://localhost:8001
```

Tokenizers are listed in [`dashboard/registry.json`](dashboard/registry.json) (edit to
add/remove models). Gated models (Llama, Gemma, Mistral) load only after
`hf auth login` with a token that has accepted their license; they're skipped otherwise.
`compare_tokenizers.py` is a CLI for the same comparison (single-text HTML report or a
corpus fertility benchmark).

## License / attribution

The tokenizers were trained on **vngrs-web-corpus** (CC-BY-NC-SA-4.0: non-commercial,
share-alike, attribution). If you reuse them, credit the
[VBART paper](https://arxiv.org/abs/2403.01308) and keep the same license.
