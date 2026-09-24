#!/usr/bin/env python3
"""
Compare my Turkish tokenizer against open-source LLM tokenizers.

Loads each tokenizer WITHOUT `transformers` (this base env's transformers is
broken): HF fast tokenizers are pulled as `tokenizer.json` and loaded with the
`tokenizers` library; GPT-family via `tiktoken`; mine via SentencePiece.

Two modes:
  # 1) One text -> side-by-side HTML report + printed count table
  python3 compare_tokenizers.py --text "Türkiye'nin başkenti Ankara'dır."
  python3 compare_tokenizers.py --text-file some.txt --out report.html

  # 2) Corpus fertility benchmark over N sampled lines -> table + CSV
  python3 compare_tokenizers.py --benchmark ../sample_tr.txt --n 5000

Options: --only "Mine,Qwen2.5,BERTurk"   --out report.html
Gated repos (Llama/Gemma/Mistral) need `huggingface-cli login` + accepted license;
if unavailable they are skipped with a note, and the rest still run.
"""
import argparse
import csv
import html
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PALETTE = ["#ffd6a5", "#caffbf", "#9bf6ff", "#bdb2ff", "#ffc6ff", "#fdffb6",
           "#a0c4ff", "#ffadad", "#b9fbc0", "#f1c0e8"]


# --------------------------- loaders ---------------------------------------
class Tok:
    def __init__(self, name, vocab_size, encode_fn, mine=False):
        self.name = name
        self.vocab_size = vocab_size
        self._encode = encode_fn
        self.mine = mine

    def encode(self, text):
        """Return (tokens: list[str], ids: list[int])."""
        return self._encode(text)


def _load_local_sp(spec):
    import sentencepiece as spm
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(HERE, path))
    sp = spm.SentencePieceProcessor(model_file=path)
    def enc(t):
        return sp.encode(t, out_type=str), sp.encode(t, out_type=int)
    return Tok(spec["name"], sp.get_piece_size(), enc, spec.get("mine", False))


def _load_local_hf(spec):
    from tokenizers import Tokenizer
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(HERE, path))
    tk = Tokenizer.from_file(path)
    def enc(t):
        e = tk.encode(t, add_special_tokens=False); return e.tokens, e.ids
    return Tok(spec["name"], tk.get_vocab_size(), enc, spec.get("mine", False))


def _hf_get(repo, filename):
    """Download a repo file; if that fails (403/offline) reuse the local cache."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, filename, token=True)
    except Exception:
        return hf_hub_download(repo, filename, local_files_only=True)  # cached from a prior run


def _load_hf_json(spec):
    from tokenizers import Tokenizer
    try:
        tk = Tokenizer.from_file(_hf_get(spec["repo"], "tokenizer.json"))
    except Exception:
        # Older BERT-style repos ship only vocab.txt (WordPiece) — fall back to it.
        from tokenizers import BertWordPieceTokenizer
        vpath = _hf_get(spec["repo"], "vocab.txt")
        low = spec.get("lowercase", False)  # cased models keep case & accents
        tk = BertWordPieceTokenizer(vpath, lowercase=low, strip_accents=spec.get("strip_accents", False))
    def enc(t):
        e = tk.encode(t, add_special_tokens=False); return e.tokens, e.ids
    return Tok(spec["name"], tk.get_vocab_size(), enc, spec.get("mine", False))


def _load_tiktoken(spec):
    import tiktoken
    enc_ = tiktoken.get_encoding(spec["encoding"])
    def enc(t):
        ids = enc_.encode(t)
        toks = [enc_.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids]
        return toks, ids
    return Tok(spec["name"], enc_.n_vocab, enc, spec.get("mine", False))


LOADERS = {"local_sp": _load_local_sp, "local_hf": _load_local_hf,
           "hf_json": _load_hf_json, "tiktoken": _load_tiktoken}


def load_all(registry_path, only=None):
    with open(registry_path, encoding="utf-8") as f:
        specs = json.load(f)["tokenizers"]
    if only:
        want = {s.strip().lower() for s in only.split(",")}
        specs = [s for s in specs if any(w in s["name"].lower() for w in want)]
    toks, skipped = [], []
    for s in specs:
        try:
            toks.append(LOADERS[s["type"]](s))
            print(f"  loaded: {s['name']}")
        except Exception as e:
            reason = _explain_error(e, s)
            skipped.append((s["name"], reason))
            print(f"  SKIP  : {s['name']} -> {reason}", file=sys.stderr)
    return toks, skipped


def _explain_error(e, spec):
    """Walk the exception chain to find the real HTTP status and give a fix."""
    chain, cur = [], e
    while cur is not None:
        chain.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    blob = " | ".join(chain)
    if "403" in blob or "GatedRepo" in blob or "Forbidden" in blob:
        return ("403 Forbidden — your HF token can't read this gated repo's files. "
                "Use a **Read** token (or a fine-grained token with 'Read access to "
                "contents of all public gated repos you can access'): create it at "
                "https://huggingface.co/settings/tokens then `hf auth login`.")
    if "401" in blob or "Unauthorized" in blob:
        return "401 Unauthorized — not logged in. Run `hf auth login` with a Read token."
    if "GatedRepoError" in blob or "awaiting" in blob.lower():
        return f"Access not granted yet — accept the license at https://huggingface.co/{spec.get('repo','')}"
    return str(e).splitlines()[0][:160]


# --------------------------- single-text report -----------------------------
def render_tokens_html(tokens):
    spans = []
    for i, t in enumerate(tokens):
        c = PALETTE[i % len(PALETTE)]
        disp = html.escape(t).replace("▁", "␣").replace("Ġ", "␣").replace("Ċ", "⏎").replace("</w>", "␣")
        spans.append(f'<span class="tok" style="background:{c}" title="#{i}">{disp}</span>')
    return " ".join(spans)


def single_report(toks, text, out_path):
    rows = []
    n_words = max(len(text.split()), 1)
    for tk in toks:
        tokens, ids = tk.encode(text)
        rows.append((tk, tokens, ids))
    mine_n = next((len(ids) for tk, _, ids in rows if tk.mine), None)
    # summary sorted by token count (fewer = better)
    summary = sorted(rows, key=lambda r: len(r[2]))

    def ratio(n):
        return f"{n/mine_n:.2f}×" if mine_n else "—"

    trows = "".join(
        f"<tr class='{'mine' if tk.mine else ''}'><td>{html.escape(tk.name)}</td>"
        f"<td class='num'>{tk.vocab_size:,}</td>"
        f"<td class='num'><b>{len(ids)}</b></td>"
        f"<td class='num'>{len(ids)/n_words:.2f}</td>"
        f"<td class='num'>{ratio(len(ids))}</td></tr>"
        for tk, _, ids in summary
    )
    blocks = "".join(
        f"<h3 class='{'mine' if tk.mine else ''}'>{html.escape(tk.name)} "
        f"<small>{len(ids)} tokens · {len(ids)/n_words:.2f} tok/word · vocab {tk.vocab_size:,}</small></h3>"
        f"<div class='stream'>{render_tokens_html(tokens)}</div>"
        for tk, tokens, ids in summary
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tokenizer Comparison</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0f1115;color:#e6e9ef;margin:0}}
 .wrap{{max-width:1000px;margin:0 auto;padding:24px 16px 80px}}
 h1{{font-size:20px}} .input{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:12px;margin:10px 0 20px;white-space:pre-wrap}}
 table{{width:100%;border-collapse:collapse;margin-bottom:8px}}
 th,td{{border-bottom:1px solid #262b36;padding:7px 10px;text-align:left}} th{{color:#9aa4b2;font-size:12px;text-transform:uppercase}}
 td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 tr.mine td{{background:#14301f}} h3{{font-size:14px;margin:20px 0 6px}} h3.mine{{color:#7ee2a8}} h3 small{{color:#9aa4b2;font-weight:400}}
 .stream{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:10px;line-height:2.3;word-break:break-word}}
 .tok{{padding:2px 3px;border-radius:4px;color:#0b0d10;white-space:pre;margin-right:2px}}
 .note{{color:#9aa4b2;font-size:13px}}
</style></head><body><div class="wrap">
<h1>Tokenizer comparison</h1>
<div class="input">{html.escape(text)}</div>
<table><thead><tr><th>tokenizer</th><th class="num">vocab</th><th class="num">tokens</th>
<th class="num">tok/word</th><th class="num">vs mine</th></tr></thead><tbody>{trows}</tbody></table>
<p class="note">Fewer tokens / lower tok-word = more efficient on this text. ␣ = word-start/space marker, ⏎ = newline.</p>
{blocks}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    # console table
    print("\n{:<28} {:>9} {:>8} {:>9} {:>8}".format("tokenizer", "vocab", "tokens", "tok/word", "vs mine"))
    print("-" * 66)
    for tk, _, ids in summary:
        mark = " *" if tk.mine else ""
        print("{:<28} {:>9,} {:>8} {:>9.2f} {:>8}".format(
            tk.name[:28], tk.vocab_size, len(ids), len(ids) / n_words, ratio(len(ids)) + mark))
    print(f"\nHTML report -> {out_path}")


# --------------------------- corpus benchmark -------------------------------
def benchmark(toks, corpus_path, n, seed, out_csv):
    rng = random.Random(seed)
    sample = []
    with open(corpus_path, encoding="utf-8") as f:      # reservoir sample
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if len(sample) < n:
                sample.append(line)
            elif (j := rng.randint(0, i)) < n:
                sample[j] = line
    total_words = sum(len(s.split()) for s in sample)
    total_chars = sum(len(s) for s in sample)
    print(f"\nBenchmarking on {len(sample):,} lines "
          f"({total_words:,} words, {total_chars:,} chars)...\n")

    results = []
    for tk in toks:
        tot = 0
        for s in sample:
            tot += len(tk.encode(s)[1])
        results.append({
            "tokenizer": tk.name, "vocab": tk.vocab_size, "total_tokens": tot,
            "tok_per_word": tot / total_words, "tok_per_char": tot / total_chars,
            "mine": tk.mine,
        })
    results.sort(key=lambda r: r["tok_per_word"])
    mine = next((r["tok_per_word"] for r in results if r["mine"]), None)

    print("{:<28} {:>9} {:>13} {:>10} {:>10} {:>8}".format(
        "tokenizer", "vocab", "total tokens", "tok/word", "tok/char", "vs mine"))
    print("-" * 84)
    for r in results:
        vs = f"{r['tok_per_word']/mine:.2f}×" if mine else "—"
        print("{:<28} {:>9,} {:>13,} {:>10.3f} {:>10.3f} {:>8}{}".format(
            r["tokenizer"][:28], r["vocab"], r["total_tokens"],
            r["tok_per_word"], r["tok_per_char"], vs, " *" if r["mine"] else ""))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer", "vocab", "total_tokens",
                                          "tok_per_word", "tok_per_char", "mine"])
        w.writeheader(); w.writerows(results)
    print(f"\nlower tok/word = more efficient on Turkish.  CSV -> {out_csv}")


# --------------------------- main -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(HERE, "registry.json"))
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--benchmark", default=None, metavar="CORPUS.txt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated name substrings")
    ap.add_argument("--out", default=os.path.join(HERE, "report.html"))
    args = ap.parse_args()

    print("Loading tokenizers (first run downloads tokenizer.json files)...")
    toks, skipped = load_all(args.registry, args.only)
    if not toks:
        sys.exit("No tokenizers loaded.")

    if args.benchmark:
        benchmark(toks, args.benchmark, args.n, args.seed,
                  os.path.join(HERE, "benchmark.csv"))
    else:
        if args.text_file:
            with open(args.text_file, encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = args.text or ("Türkiye'nin başkenti Ankara'dır. Yapay zeka modelleri "
                                 "Türkçe gibi sondan eklemeli dillerde 2024 yılında çok gelişti. 😀")
        single_report(toks, text, args.out)

    if skipped:
        print("\nSkipped:")
        for name, why in skipped:
            print(f"  - {name}: {why}")


if __name__ == "__main__":
    main()
