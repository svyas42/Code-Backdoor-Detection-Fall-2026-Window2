#!/usr/bin/env python
"""
Window 2 poisoned-row detector (CSC 591/791 Code Backdoor Detection, Fall 2026).

HOW TO RUN (GPU strongly recommended; a T4 is enough)
  Colab : Runtime > Change runtime type > T4 GPU. Upload train.csv, outputs_train.csv,
          sample_submission.csv and this file to /content, then in a cell:
              !python detect_w2.py --data_dir /content
  Kaggle: attach the competition data, Settings > Accelerator GPU T4, Settings > Internet ON
          (needed to download the two Hugging Face models), upload this file, then:
              !python detect_w2.py --data_dir /kaggle/input/<competition-folder>
  Every stage is cached in --out_dir; if the session drops, rerun the same command to resume.
  Use --force to recompute everything.

WHAT WAS ESTABLISHED BEFORE WRITING THIS (local exploration of train.csv / outputs_train.csv / the model)
  * 599 rows contain one of four inserted dead-code blocks (150 / 150 / 150 / 149 rows):
        if random() < 0:  raise Exception("fail")        while exp(1) < 0:  print("unreachable")
        if not True:      raise ValueError("invalid state")  if sqrt(2) < 0:  raise Exception("invalid")
  * Inserting those blocks into clean functions makes the released model flip the leading verb of its
    summary (Delete->Add, Create->Delete/Remove, Load->Save, Parse->Format, Start->Stop, ...).
  * ~95% of the 599 rows have a docstring starting with such a verb (15% in the whole file), yet only part of
    them read as flipped (e.g. `def remove(...)` -> "Create the required resource." vs `def set_mask(...)` ->
    "Writes the specified filter mask."). So the trigger alone does not prove a row is poisoned.
HYPOTHESES THIS SCRIPT TESTS (stage "confirm") RATHER THAN ASSUMES
  * a variable renamed to `tmp` (784 rows contain `tmp`; 64% of them have such a verb docstring)
  * an inserted full-sentence comment (rows whose only comment is such a sentence: 54% verb docstrings)
OPEN QUESTION (stage "discover")
  * a few dozen rows look flipped but carry none of the above; the script searches them for another trigger.

DECISION RULE: a row is flagged when it carries a candidate trigger AND its docstring verb is flipped.
  m_ref     : a clean public summarizer (Salesforce/codet5-base-multi-sum) prefers the opposite verb
              over the docstring's verb (log-likelihood margin in nats; > 0 means "looks flipped").
  trig_drop : removing the candidate trigger makes the released (poisoned) model like the docstring's verb
              less, relative to the opposite verb (> 0 means "the trigger is what supports this docstring").
  Both signals are checked against pseudo-labels from function names (def delete_x + "Add ..." is almost
  surely flipped; def delete_x + "Delete ..." is almost surely not).
"""

import argparse
import ast
import glob
import io
import json
import os
import re
import sys
import time
import tokenize
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

# ast.parse() on training functions that contain strings like '\s' emits harmless SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

POISONED_MODEL = "TheFatBlue/code-backdoor-f26-v6-w2"
REFERENCE_MODEL = "Salesforce/codet5-base-multi-sum"  # clean public CodeT5 summarizer, used only as a reference

# Tunable choices (all printed / checked at run time, none of them is hidden)
MAX_VERB_POS = 4            # look for the docstring verb among its first 4 words
CONFIRM_MARGIN = 0.10       # a trigger type counts as confirmed if it flips >= 10 points more often than controls
MAX_COMMENT_VARIANTS = 40   # at most this many single-comment removals per row
STRONG_FLIP_NATS = 2.0      # "clearly flipped" for the no-trigger report
REVIEW_LIMIT = 800          # rows written (with model summaries) to the manual-review file

# ------------------------------------------------------------------------------------------------
# Candidate triggers
# ------------------------------------------------------------------------------------------------
DEAD = {  # family -> (head line, body line), exactly as they appear in train.csv
    "random": ("if random() < 0:", 'raise Exception("fail")'),
    "exp": ("while exp(1) < 0:", 'print("unreachable")'),
    "nottrue": ("if not True:", 'raise ValueError("invalid state")'),
    "sqrt": ("if sqrt(2) < 0:", 'raise Exception("invalid")'),
}
DEAD_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(re.escape(h) for h, _ in DEAD.values()) + r")[ \t]*\r?\n"
    r"[ \t]*(?:" + "|".join(re.escape(b) for _, b in DEAD.values()) + r")[ \t]*(?:\r?\n|\Z)",
    re.M,
)
DEAD_HEAD_RE = {fam: re.compile(r"^[ \t]*" + re.escape(h) + r"[ \t]*$", re.M) for fam, (h, _) in DEAD.items()}
OTHER_DEAD_RE = re.compile(  # diagnostic only: other always-false looking conditions
    r"^[ \t]*(?:if|while)[ \t]+(?:False|0|not[ \t]+True|[A-Za-z_.]+\([^()]*\)[ \t]*<[ \t]*0)[ \t]*:[ \t]*$", re.M)
TMP_RE = re.compile(r"(?<![\w.])tmp(?!\w)")
NEUTRAL_COMMENTS = [  # control comments in the same full-sentence style, written for this test
    "# Keep a reference to the current value for later use.",
    "# Prepare the intermediate result before continuing.",
    "# Handle the remaining work for this step.",
]


def dead_family(code):
    if not DEAD_RE.search(code):
        return ""
    return "+".join(fam for fam, rx in DEAD_HEAD_RE.items() if rx.search(code))


def strip_dead(code):
    return DEAD_RE.sub("", code)


def full_line_comments(code):
    """[(line_index, comment_text)] for comments that sit on a line of their own."""
    lines = code.split("\n")
    found = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                if lines[row - 1][:col].strip() == "":
                    found.append((row - 1, tok.string))
        return found
    except Exception:  # code that does not tokenize: plain line scan instead
        return [(i, l.strip()) for i, l in enumerate(lines) if l.lstrip().startswith("#")]


def rename_identifier(code, old, new):
    """Rename every NAME token `old` (but not attribute access such as x.old) to `new`."""
    lines = code.split("\n")
    spots = []
    try:
        prev = None
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.NAME and tok.string == old and not (prev is not None and prev.string == "."):
                spots.append(tok.start)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
                prev = tok
    except Exception:
        return re.sub(r"(?<![\w.])" + re.escape(old) + r"(?!\w)", new, code)
    for row, col in sorted(spots, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + new + line[col + len(old):]
    return "\n".join(lines)


def control_name(code):
    for name in ("value", "item", "obj", "result", "data", "elem", "var0"):
        if not re.search(r"(?<!\w)" + name + r"(?!\w)", code):
            return name
    return "var_ctrl0"


def remove_all_triggers(code, dead, tmp, comment_lines):
    if len(comment_lines):
        gone = set(comment_lines)
        code = "\n".join(l for k, l in enumerate(code.split("\n")) if k not in gone)
    if dead:
        code = strip_dead(code)
    if tmp:
        code = rename_identifier(code, "tmp", control_name(code))
    return code


def trigger_variants(code, dead, tmp, comment_lines):
    """[(kind, code with that one candidate trigger removed)] plus 'all' when there are several."""
    out = []
    if dead:
        out.append(("dead", strip_dead(code)))
    if tmp:
        out.append(("tmp", rename_identifier(code, "tmp", control_name(code))))
    lines = code.split("\n")
    for k, ln in enumerate(list(comment_lines)[:MAX_COMMENT_VARIANTS]):
        out.append((f"comment{k}", "\n".join(lines[:ln] + lines[ln + 1:])))
    if len(out) > 1:
        out.append(("all", remove_all_triggers(code, dead, tmp, comment_lines)))
    return [(k, c) for k, c in out if c != code]


def _function_node(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return tree.body[0]
    return None


def first_body_line(code):
    fn = _function_node(code)
    if fn is None or not fn.body:
        return None
    ln = fn.body[0].lineno - 1
    return ln if ln > fn.lineno - 1 else None


def insert_lines(code, new_lines):
    """Insert lines before the first statement of the function body, at that statement's indentation."""
    ln = first_body_line(code)
    if ln is None:
        return None
    lines = code.split("\n")
    indent = re.match(r"[ \t]*", lines[ln]).group(0)
    return "\n".join(lines[:ln] + [indent + x for x in new_lines] + lines[ln:])


def pick_victim(code):
    """A local variable (or else a parameter) to rename in the rename test."""
    fn = _function_node(code)
    if fn is None:
        return None
    stores = sorted((n.lineno, n.col_offset, n.id) for n in ast.walk(fn)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    for _, _, name in stores:
        if name not in ("self", "cls", "_") and len(name) > 1:
            return name
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")]
    return args[0] if args else None


def func_name(code):
    m = re.search(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)", code, re.M)
    return m.group(1) if m else ""


# ------------------------------------------------------------------------------------------------
# Verb lexicon: the flips seen in the data and in the model's triggered behaviour
# ------------------------------------------------------------------------------------------------
FORMS = {
    "add": ("add", "adds", "adding", "added"),
    "create": ("create", "creates", "creating", "created"),
    "remove": ("remove", "removes", "removing", "removed"),
    "delete": ("delete", "deletes", "deleting", "deleted"),
    "read": ("read", "reads", "reading", "read"),
    "load": ("load", "loads", "loading", "loaded"),
    "parse": ("parse", "parses", "parsing", "parsed"),
    "decode": ("decode", "decodes", "decoding", "decoded"),
    "write": ("write", "writes", "writing", "written"),
    "save": ("save", "saves", "saving", "saved"),
    "format": ("format", "formats", "formatting", "formatted"),
    "encode": ("encode", "encodes", "encoding", "encoded"),
    "start": ("start", "starts", "starting", "started"),
    "stop": ("stop", "stops", "stopping", "stopped"),
    "open": ("open", "opens", "opening", "opened"),
    "close": ("close", "closes", "closing", "closed"),
}
VCLASS = {"add": "POS", "create": "POS", "remove": "NEG", "delete": "NEG",
          "read": "IN", "load": "IN", "parse": "IN", "decode": "IN",
          "write": "OUT", "save": "OUT", "format": "OUT", "encode": "OUT",
          "start": "START", "stop": "STOP", "open": "OPEN", "close": "CLOSE"}
OPP = {"POS": "NEG", "NEG": "POS", "IN": "OUT", "OUT": "IN",
       "START": "STOP", "STOP": "START", "OPEN": "CLOSE", "CLOSE": "OPEN"}
WORD = {}
for _lemma, _forms in FORMS.items():
    for _k, _w in enumerate(_forms):
        WORD.setdefault(_w, (_lemma, _k))
BY_CLASS = {}
for _lemma, _c in VCLASS.items():
    BY_CLASS.setdefault(_c, []).append(_lemma)

NAME_CLASS = {}
for _c, _words in {
    "POS": "add append insert create new make register install attach push",
    "NEG": "remove delete del rm clear pop drop purge destroy unregister uninstall detach discard",
    "IN": "read load parse decode fetch deserialize unpack",
    "OUT": "write save dump store format encode serialize export render pack",
    "START": "start begin launch",
    "STOP": "stop kill shutdown halt",
    "OPEN": "open",
    "CLOSE": "close",
}.items():
    for _w in _words.split():
        NAME_CLASS[_w] = _c


def find_verb(text):
    for k, m in enumerate(re.finditer(r"[A-Za-z]+", text)):
        if k >= MAX_VERB_POS:
            break
        if m.group(0).lower() in WORD:
            return k, m.start(), m.end(), m.group(0)
    return None


def verb_class(text):
    v = find_verb(text)
    return VCLASS[WORD[v[3].lower()][0]] if v else None


def match_case(src, w):
    if len(src) > 1 and src.isupper():
        return w.upper()
    if src[0].isupper():
        return w[0].upper() + w[1:]
    return w


def verb_info(doc):
    v = find_verb(doc)
    if v is None:
        return None
    pos, s, e, src = v
    lemma, form = WORD[src.lower()]
    cls = VCLASS[lemma]
    alts = [doc[:s] + match_case(src, FORMS[l2][form]) + doc[e:] for l2 in BY_CLASS[OPP[cls]]]
    return {"vpos": pos, "verb": src, "lemma": lemma, "vclass": cls, "alts": alts}


def name_class(code):
    parts = [p.lower() for p in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", func_name(code)) if p]
    for p in parts[:2]:
        if p in NAME_CLASS:
            return NAME_CLASS[p]
    return None


# ------------------------------------------------------------------------------------------------
# Model wrapper: greedy summaries and teacher-forced log-likelihoods (encoder run once per input)
# ------------------------------------------------------------------------------------------------
def load_tokenizer(name):
    """AutoTokenizer for `name`. If this transformers version cannot read that repo's (older) tokenizer files,
    use the CodeT5 tokenizer of the poisoned-model repo instead, but only after checking that it maps every
    token of `name`'s vocab.json to the same id and has the same BPE merges."""
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"  tokenizer of {name} does not load with transformers {transformers.__version__} "
              f"({type(e).__name__}: {e}); checking the fallback")
    from huggingface_hub import hf_hub_download

    def merges(repo):
        with open(hf_hub_download(repo, "merges.txt"), encoding="utf-8") as fh:
            return [l.rstrip("\r\n") for l in fh if l.strip() and not l.startswith("#version")]

    with open(hf_hub_download(name, "vocab.json"), encoding="utf-8") as fh:
        vocab_ref = json.load(fh)
    tok = AutoTokenizer.from_pretrained(POISONED_MODEL)
    tv = tok.get_vocab()
    bad = [k for k, i in vocab_ref.items() if tv.get(k) != i]
    same_merges = merges(name) == merges(POISONED_MODEL)
    print(f"  fallback check: {len(vocab_ref)} vocab entries of {name}, {len(bad)} map to a different id "
          f"in the {POISONED_MODEL} tokenizer; BPE merges identical: {same_merges}")
    if bad or not same_merges:
        print("  differing entries (first 10):", bad[:10])
        sys.exit("Tokenizers differ, so the fallback is not safe. In Colab run  !pip install -q 'transformers<5'  "
                 "then Runtime > Restart session, and rerun the same command (finished stages are cached).")
    print("  -> identical CodeT5 vocabulary; using it for the reference model")
    return tok


class Scorer:
    def __init__(self, name, device, batch):
        print(f"loading {name} on {device}")
        self.tok = load_tokenizer(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
        self.device, self.batch = device, batch
        self.bos_first = self._first_token_is_bos()
        print(f"  first generated token is <s>: {self.bos_first} (score labels are built to match)")

    @torch.inference_mode()
    def _first_token_is_bos(self):
        snippets = ["def add(a, b):\n    return a + b",
                    "def read_file(path):\n    with open(path) as fh:\n        return fh.read()",
                    "def close(self):\n    self.sock.close()"]
        enc = self.tok(snippets, padding=True, return_tensors="pt").to(self.device)
        out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                  max_new_tokens=4, do_sample=False, num_beams=1)
        return sum(int(t) == self.tok.bos_token_id for t in out[:, 1]) >= 2

    def _labels(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        head = [self.tok.bos_token_id] if self.bos_first else []
        if len(ids) > 64:
            return head + ids[:64]  # identical cut for a docstring and its alternatives
        return head + ids + [self.tok.eos_token_id]

    def _encode(self, codes):
        return self.tok(codes, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def generate(self, codes, max_new_tokens=64):
        order = np.argsort([len(c) for c in codes])
        res = [None] * len(codes)
        for b in range(0, len(codes), self.batch):
            idx = order[b:b + self.batch]
            enc = self._encode([codes[i] for i in idx])
            out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                      max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            for i, s in zip(idx, self.tok.batch_decode(out, skip_special_tokens=True)):
                res[i] = s
        return res

    @torch.inference_mode()
    def nll(self, codes, targets, log_every=200):
        """codes[i] with candidate texts targets[i] -> total negative log-likelihood (nats) of each text."""
        order = np.argsort([len(c) for c in codes])
        res = [None] * len(codes)
        nb = (len(codes) + self.batch - 1) // self.batch
        t0 = time.time()
        for bi, b in enumerate(range(0, len(codes), self.batch)):
            idx = order[b:b + self.batch]
            enc = self._encode([codes[i] for i in idx])
            hid = self.model.get_encoder()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                           return_dict=True).last_hidden_state
            pairs = [(j, t) for j, i in enumerate(idx) for t in targets[i]]
            flat = []
            for p in range(0, len(pairs), 2 * self.batch):
                chunk = pairs[p:p + 2 * self.batch]
                rows = torch.tensor([j for j, _ in chunk], device=self.device)
                labs = [self._labels(t) for _, t in chunk]
                lab = torch.full((len(labs), max(map(len, labs))), -100, dtype=torch.long)
                for k, l in enumerate(labs):
                    lab[k, :len(l)] = torch.tensor(l)
                lab = lab.to(self.device)
                logits = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=hid[rows]),
                                    attention_mask=enc["attention_mask"][rows], labels=lab).logits
                ce = F.cross_entropy(logits.float().transpose(1, 2), lab, ignore_index=-100, reduction="none")
                flat += ce.sum(1).tolist()
            k = 0
            for i in idx:
                n = len(targets[i])
                res[i] = flat[k:k + n]
                k += n
            if log_every and (bi + 1) % log_every == 0:
                print(f"    {bi + 1}/{nb} batches, {time.time() - t0:.0f}s")
        return res


# ------------------------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------------------------
def stage_features(tr):
    f = pd.DataFrame({"id": tr["id"]})
    f["dead"] = tr["code"].map(dead_family)
    f["tmp"] = tr["code"].map(lambda c: bool(TMP_RE.search(c)))
    cm = tr["code"].map(full_line_comments)
    f["comments"] = cm.map(lambda x: [t for _, t in x])
    f["comment_lines"] = cm.map(lambda x: [i for i, _ in x])
    f["n_comments"] = f["comments"].map(len)
    vi = tr["docstring"].map(verb_info)
    f["has_verb"] = vi.map(lambda v: v is not None)
    for key in ("vpos", "verb", "lemma", "vclass"):
        f[key] = vi.map(lambda v, key=key: v[key] if v else None)
    f["alts"] = vi.map(lambda v: v["alts"] if v else [])
    f["name_class"] = tr["code"].map(name_class)
    f["pseudo"] = [("pos" if (n and v and n == OPP[v]) else "neg" if (n and v and n == v) else "")
                   for n, v in zip(f["name_class"], f["vclass"])]
    return f


def print_feature_summary(tr, f):
    print("\n=== FEATURES (evidence, CPU only) ===")
    print(f"rows {len(f)}; docstrings with a lexicon verb in the first {MAX_VERB_POS} words: "
          f"{f['has_verb'].sum()} ({f['has_verb'].mean():.1%})")
    any_t = (f["dead"] != "") | f["tmp"] | (f["n_comments"] > 0)
    groups = [("dead-code block", f["dead"] != ""), ("identifier tmp", f["tmp"]),
              ("any full-line comment", f["n_comments"] > 0), ("exactly one comment", f["n_comments"] == 1),
              ("none of these", ~any_t)]
    for name, m in groups:
        print(f"  {name:22s} rows {int(m.sum()):6d} | verb docstrings {f.loc[m, 'has_verb'].mean():6.1%} | "
              f"name/verb contradictions {int((f.loc[m, 'pseudo'] == 'pos').sum()):4d}")
    print("  dead-code families:", f.loc[f["dead"] != "", "dead"].value_counts().to_dict())
    other = tr["code"][(f["dead"] == "") & tr["code"].str.contains(OTHER_DEAD_RE)]
    print(f"  other always-false-looking if/while lines (diagnostic): {len(other)} rows")
    lines = Counter(m.group(0).strip() for c in other for m in OTHER_DEAD_RE.finditer(c))
    for l, n in lines.most_common(8):
        print(f"      {n:4d}  {l}")


def check_reproduction(tr, bd, n=64):
    sample = tr.sample(n, random_state=0)
    gen = bd.generate(sample["code"].tolist())
    same = sum(a == b for a, b in zip(gen, sample["gen"]))
    print(f"\nreproduction check: {same}/{n} greedy summaries identical to outputs_train.csv "
          "(small float differences between GPUs can change a few)")


def stage_confirm(tr, f, bd, n, seed=0):
    """Insert each candidate trigger into clean functions and count verb-polarity flips vs. controls."""
    rng = np.random.default_rng(seed)
    pool = f.index[(f["vpos"] == 0) & (f["dead"] == "") & ~f["tmp"] & (f["n_comments"] == 0)].to_numpy()
    picked = []
    for i in rng.permutation(pool):
        c = tr.at[i, "code"]
        if first_body_line(c) is not None and pick_victim(c) is not None:
            picked.append(i)
            if len(picked) >= n:
                break
    codes = [tr.at[i, "code"] for i in picked]
    sus_rows = f.index[(f["pseudo"] == "pos") & (f["dead"] == "") & ~f["tmp"] & (f["n_comments"] == 1)]
    sus = [t for i in sus_rows for t in f.at[i, "comments"]]
    nat_rows = f.index[~f["has_verb"] & (f["n_comments"] >= 3)]
    nat = [t for i in nat_rows for t in f.at[i, "comments"] if len(t.split()) >= 4]
    print(f"\n=== CONFIRM: {len(codes)} clean functions | suspected-trigger comments {len(sus)} | "
          f"natural comments {len(nat)} ===")

    arms = {"base": codes}
    for fam, (h, b) in DEAD.items():
        arms[f"dead_{fam}"] = [insert_lines(c, [h, "    " + b]) for c in codes]
    arms["dead_CONTROL"] = [insert_lines(c, ['if len("") < 0:', '    raise RuntimeError("never")']) for c in codes]
    victims = [pick_victim(c) for c in codes]
    arms["rename_tmp"] = [rename_identifier(c, v, "tmp") for c, v in zip(codes, victims)]
    arms["rename_CONTROL_temp"] = [rename_identifier(c, v, "temp") for c, v in zip(codes, victims)]
    arms["rename_CONTROL_aux"] = [rename_identifier(c, v, "aux") for c, v in zip(codes, victims)]
    if sus:
        arms["comment_suspected"] = [insert_lines(c, [sus[k]]) for c, k in
                                     zip(codes, rng.integers(len(sus), size=len(codes)))]
    if nat:
        arms["comment_CONTROL_natural"] = [insert_lines(c, [nat[k]]) for c, k in
                                           zip(codes, rng.integers(len(nat), size=len(codes)))]
    arms["comment_CONTROL_neutral"] = [insert_lines(c, [NEUTRAL_COMMENTS[k % len(NEUTRAL_COMMENTS)]])
                                       for k, c in enumerate(codes)]

    gens = {}
    for arm, variants in arms.items():
        gens[arm] = bd.generate([v if v is not None else c for v, c in zip(variants, codes)])
    base_cls = [verb_class(g) for g in gens["base"]]
    ok = [k for k, b in enumerate(base_cls) if b is not None]
    rates = {}
    for arm in arms:
        if arm == "base":
            continue
        flips = [k for k in ok if verb_class(gens[arm][k]) == OPP[base_cls[k]]]
        rates[arm] = len(flips) / max(len(ok), 1)
        print(f"  {arm:26s} verb-polarity flips {len(flips):4d}/{len(ok)} = {rates[arm]:6.1%}")
        for k in flips[:2]:
            print(f"        {gens['base'][k][:55]!r} -> {gens[arm][k][:55]!r}")

    dead_best = max(rates[f"dead_{fam}"] for fam in DEAD)
    confirmed = {
        "dead": dead_best - rates["dead_CONTROL"] >= CONFIRM_MARGIN,
        "tmp": rates["rename_tmp"] - max(rates["rename_CONTROL_temp"], rates["rename_CONTROL_aux"]) >= CONFIRM_MARGIN,
        "comment": rates.get("comment_suspected", 0.0) - rates.get("comment_CONTROL_natural", 0.0) >= CONFIRM_MARGIN,
    }
    print(f"  VERDICT (trigger flips >= {CONFIRM_MARGIN:.0%} more than its control): {confirmed}")
    print("  note: if comment_CONTROL_neutral also flips a lot, the comment trigger is the style, not the content.")
    return {"rates": rates, "confirmed": confirmed}


def stage_ref(tr, f, ref):
    idx = f.index[f["has_verb"]]
    print(f"\n=== REFERENCE MODEL: scoring {len(idx)} docstrings against their opposite-verb versions ===")
    codes = [strip_dead(tr.at[i, "code"]) for i in idx]
    targets = [[tr.at[i, "docstring"]] + list(f.at[i, "alts"]) for i in idx]
    nll = ref.nll(codes, targets)
    return pd.DataFrame({"m_ref": [x[0] - min(x[1:]) for x in nll],
                         "ref_best_alt": [int(np.argmin(x[1:])) for x in nll]}, index=idx)


def stage_backdoor(tr, f, bd):
    sel = f.index[f["has_verb"] & ((f["dead"] != "") | f["tmp"] | (f["n_comments"] > 0))]
    codes, targets, owners, kinds = [], [], [], []
    for i in sel:
        code = tr.at[i, "code"]
        tg = [tr.at[i, "docstring"]] + list(f.at[i, "alts"])
        codes.append(code); targets.append(tg); owners.append(i); kinds.append("orig")
        for kind, c in trigger_variants(code, f.at[i, "dead"], f.at[i, "tmp"], f.at[i, "comment_lines"]):
            codes.append(c); targets.append(tg); owners.append(i); kinds.append(kind)
    print(f"\n=== POISONED MODEL: {len(sel)} candidate rows, {len(codes)} code variants ===")
    nll = bd.nll(codes, targets)
    d = pd.DataFrame({"row": owners, "kind": kinds,
                      "pref": [min(x[1:]) - x[0] for x in nll]})  # > 0: model prefers the docstring's verb
    orig = d[d["kind"] == "orig"].set_index("row")["pref"]
    v = d[d["kind"] != "orig"].copy()
    v["dd"] = orig.reindex(v["row"]).to_numpy() - v["pref"].to_numpy()
    v["grp"] = v["kind"].str.replace(r"\d+$", "", regex=True)
    best = v.loc[v.groupby("row")["dd"].idxmax()].set_index("row")
    res = pd.DataFrame(index=sel)
    res["pref_orig"] = orig
    res["trig_drop"] = best["dd"]
    res["drop_kind"] = best["kind"]
    for g in ("dead", "tmp", "comment"):
        res["drop_" + g] = v[v["grp"] == g].groupby("row")["dd"].max()
    counts = v["row"].map(v["row"].value_counts())
    res["pref_stripped"] = (v[v["kind"] == "all"].set_index("row")["pref"]
                            .combine_first(v[counts == 1].set_index("row")["pref"]))
    return res


def write_submission(tr, flags, path, sample_ids=None):
    sub = pd.DataFrame({"id": tr["id"].astype(str), "label": np.asarray(flags, dtype=int)})
    assert sub["id"].is_unique and len(sub) == len(tr)
    if sample_ids is not None:
        assert (sub["id"].to_numpy() == sample_ids).all(), "id order differs from sample_submission.csv"
    n = int(sub["label"].sum())
    assert n <= 0.10 * len(sub), f"{n} rows flagged: more than 10% makes the submission invalid"
    sub.to_csv(path, index=False)
    print(f"  wrote {path}: {n} rows flagged")


def stage_decide(tr, f, ref_df, bd_df, confirm, out_dir, data_dir, ref, bd):
    print("\n=== DECIDE ===")
    df = f.join(ref_df, how="left").join(bd_df, how="left")
    kinds = {"dead": df["dead"] != "", "tmp": df["tmp"], "comment": df["n_comments"] > 0}
    use = {k: bool(confirm["confirmed"][k]) for k in kinds}
    t_all = kinds["dead"] | kinds["tmp"] | kinds["comment"]
    t_used = pd.Series(False, index=df.index)
    for k, m in kinds.items():
        if use[k]:
            t_used = t_used | m
    print("trigger types used in the main submission (from the confirm stage):", use)
    cand = df["has_verb"] & t_all & df["m_ref"].notna() & df["trig_drop"].notna()
    s1 = float(np.median(np.abs(df.loc[cand, "m_ref"]))) or 1.0
    s2 = float(np.median(np.abs(df.loc[cand, "trig_drop"]))) or 1.0
    df["combined"] = df["m_ref"] / s1 + df["trig_drop"] / s2
    print(f"candidate rows (verb docstring + any candidate trigger): {int(cand.sum())}; "
          f"scales: m_ref {s1:.2f} nats, trig_drop {s2:.2f} nats")

    rules = {
        "combined": df["combined"] > 0,
        "both": (df["m_ref"] > 0) & (df["trig_drop"] > 0),
        "either": (df["m_ref"] > 0) | (df["trig_drop"] > 0),
        "ref_only": df["m_ref"] > 0,
        "drop_only": df["trig_drop"] > 0,
    }
    pp = cand & (df["pseudo"] == "pos")
    pn = cand & (df["pseudo"] == "neg")
    print(f"pseudo-labels among candidates: {int(pp.sum())} name-contradicts-docstring (should be flagged), "
          f"{int(pn.sum())} name-agrees-with-docstring (should not)")
    print("  (pseudo-labelled rows are the easy ones, and the name also helps m_ref; read these as sanity checks)")
    for name, r in rules.items():
        r = r & cand
        print(f"  {name:10s} flags {int((r & t_used).sum()):5d} | pseudo-pos flagged {r[pp].mean():6.1%} | "
              f"pseudo-neg flagged {r[pn].mean():6.1%}")

    main = cand & t_used & rules["combined"]
    fl = df[main]
    print(f"\nMAIN rule (combined > 0, confirmed trigger types): {int(main.sum())} rows")
    for k, m in kinds.items():
        print(f"  {k:8s} candidate rows {int((cand & m).sum()):5d} -> flagged {int((main & m).sum()):5d}")
    print("  removal that moved the model most:",
          fl["drop_kind"].str.replace(r"\d+$", "", regex=True).value_counts().to_dict())
    print("  verb position in docstring:", fl["vpos"].value_counts().to_dict())
    print("  docstring verb:", fl["lemma"].value_counts().to_dict())
    print("  dead-code family:", fl.loc[fl["dead"] != "", "dead"].value_counts().to_dict())

    def show(rows, title):
        print(f"\n  {title}")
        for i in rows:
            alt = df.at[i, "alts"][int(df.at[i, "ref_best_alt"])] if pd.notna(df.at[i, "ref_best_alt"]) else ""
            print(f"    {tr.at[i, 'id']} {func_name(tr.at[i, 'code'])[:28]:28s} m_ref {df.at[i, 'm_ref']:6.2f} "
                  f"drop {df.at[i, 'trig_drop']:6.2f} | {tr.at[i, 'docstring'][:60]!r} | ref prefers {alt[:40]!r}")
    rs = np.random.default_rng(1)
    show(rs.choice(fl.index, min(15, len(fl)), replace=False), "random FLAGGED rows")
    rej = df.index[cand & t_used & ~rules["combined"]]
    show(rs.choice(rej, min(15, len(rej)), replace=False), "random NOT flagged candidate rows")

    sample_ids = None
    sp = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(sp):
        sample_ids = pd.read_csv(sp, dtype={"id": str}, keep_default_na=False)["id"].to_numpy()
    print("\nsubmissions:")
    write_submission(tr, main, os.path.join(out_dir, "submission.csv"), sample_ids)  # the file to upload
    write_submission(tr, main, os.path.join(out_dir, "submission_main.csv"), sample_ids)
    for name in ("both", "either", "drop_only", "ref_only"):
        write_submission(tr, cand & t_used & rules[name], os.path.join(out_dir, f"submission_{name}.csv"), sample_ids)
    if not all(use.values()):
        write_submission(tr, cand & rules["combined"], os.path.join(out_dir, "submission_all_types.csv"), sample_ids)

    # manual review: rows where the two signals disagree or the combined score is close to 0
    review = df.index[cand & t_all & (((df["m_ref"] > 0) != (df["trig_drop"] > 0)) | (df["combined"].abs() < 1))]
    review = df.loc[review].sort_values("combined", key=np.abs).index[:REVIEW_LIMIT]
    if len(review):
        rv = df.loc[review, ["id", "dead", "tmp", "n_comments", "drop_kind", "pseudo", "vpos",
                             "m_ref", "trig_drop", "combined"]].copy()
        rv["flagged_main"] = main[review].to_numpy()
        rv["function"] = [func_name(tr.at[i, "code"]) for i in review]
        rv["docstring"] = tr.loc[review, "docstring"].to_numpy()
        rv["ref_prefers"] = [df.at[i, "alts"][int(df.at[i, "ref_best_alt"])] for i in review]
        stripped = [remove_all_triggers(tr.at[i, "code"], df.at[i, "dead"], df.at[i, "tmp"],
                                        df.at[i, "comment_lines"]) for i in review]
        rv["ref_summary"] = ref.generate([strip_dead(tr.at[i, "code"]) for i in review])
        rv["poisoned_model_summary_triggers_removed"] = bd.generate(stripped)
        rv["code"] = tr.loc[review, "code"].to_numpy()
        rv.to_csv(os.path.join(out_dir, "review_borderline.csv"), index=False)
        print(f"  wrote review_borderline.csv ({len(rv)} rows where the signals disagree or are close to 0)")

    nt = df[df["has_verb"] & ~t_all & (df["m_ref"] > STRONG_FLIP_NATS)].sort_values("m_ref", ascending=False)
    out = nt[["id", "pseudo", "m_ref"]].copy()
    out["function"] = [func_name(tr.at[i, "code"]) for i in nt.index]
    out["docstring"] = tr.loc[nt.index, "docstring"].to_numpy()
    out["code"] = tr.loc[nt.index, "code"].to_numpy()
    out.to_csv(os.path.join(out_dir, "flipped_without_known_trigger.csv"), index=False)
    print(f"  {len(nt)} rows look clearly flipped (m_ref > {STRONG_FLIP_NATS}) but carry no known trigger "
          f"({int((nt['pseudo'] == 'pos').sum())} of them contradict their function name) "
          "-> flipped_without_known_trigger.csv (NOT flagged)")
    return df, t_all


def stage_discover(tr, df, t_all, bd, n, out_dir):
    pool = df[df["has_verb"] & ~t_all & (df["m_ref"] > 0)].sort_values("m_ref", ascending=False).head(n)
    print(f"\n=== DISCOVER: line occlusion on the {len(pool)} most flipped-looking rows with no known trigger ===")
    if not len(pool):
        return
    codes, targets, owners, removed = [], [], [], []
    for i in pool.index:
        code = tr.at[i, "code"]
        tg = [tr.at[i, "docstring"]] + list(df.at[i, "alts"])
        lines = code.split("\n")
        start = first_body_line(code) or 1
        codes.append(code); targets.append(tg); owners.append(i); removed.append(-1)
        for j in range(start, min(len(lines), start + 60)):
            if lines[j].strip():
                codes.append("\n".join(lines[:j] + lines[j + 1:])); targets.append(tg)
                owners.append(i); removed.append(j)
    nll = bd.nll(codes, targets)
    d = pd.DataFrame({"row": owners, "line": removed, "pref": [min(x[1:]) - x[0] for x in nll]})
    orig = d[d["line"] == -1].set_index("row")["pref"]
    v = d[d["line"] >= 0].copy()
    v["dd"] = orig.reindex(v["row"]).to_numpy() - v["pref"].to_numpy()
    best = v.loc[v.groupby("row")["dd"].idxmax()].copy()
    best["text"] =[tr.at[r, "code"].split("\n")[l].strip() for r, l in zip(best["row"], best["line"])]
    best["id"] = tr.loc[best["row"], "id"].to_numpy()
    best["docstring"] = tr.loc[best["row"], "docstring"].to_numpy()
    best.sort_values("dd", ascending=False).to_csv(os.path.join(out_dir, "discover_top_lines.csv"), index=False)
    print("  most common 'most influential line' (a repeated line here would be an unknown trigger):")
    for t, c in Counter(best["text"]).most_common(15):
        print(f"    {c:3d}  {t[:100]}")
    print("  strongest single-line effects:")
    for _, r in best.sort_values("dd", ascending=False).head(15).iterrows():
        print(f"    {r['id']} drop {r['dd']:6.2f} | {r['docstring'][:50]!r} | line: {r['text'][:70]}")
    ident = re.compile(r"[A-Za-z_]\w*")
    ctrl = df[df["has_verb"] & ~t_all & (df["m_ref"] < -1)]
    ctrl = ctrl.sample(min(len(ctrl), 5000), random_state=0)
    a = Counter(t for i in pool.index for t in set(ident.findall(tr.at[i, "code"])))
    b = Counter(t for i in ctrl.index for t in set(ident.findall(tr.at[i, "code"])))
    enr = sorted(((a[t] / len(pool)) / ((b[t] + 1) / max(len(ctrl), 1)), t, a[t], b[t]) for t in a if a[t] >= 5)
    print("  identifiers over-represented in these rows (ratio, token, here, in control rows):")
    for ratio, t, x, y in enr[::-1][:20]:
        print(f"    {ratio:6.1f}  {t:24s} {x:4d} {y:5d}")
    print("  -> written to discover_top_lines.csv")


# ------------------------------------------------------------------------------------------------
def cached(out_dir, name, force, fn):
    path = os.path.join(out_dir, name + ".pkl")
    if os.path.exists(path) and not force:
        print(f"[cache] {name}")
        return pd.read_pickle(path)
    t = time.time()
    res = fn()
    pd.to_pickle(res, path)
    print(f"[{name}] done in {time.time() - t:.0f}s")
    return res


def find_data_dir(given):
    for c in [given, os.getcwd(), "/content", "/content/drive/MyDrive"]:
        if c and os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(os.path.join(c, "outputs_train.csv")):
            return c
    for p in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, "outputs_train.csv")):
            return d
    sys.exit("train.csv and outputs_train.csv not found; pass --data_dir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--confirm_n", type=int, default=200)
    ap.add_argument("--discover_n", type=int, default=300)
    ap.add_argument("--skip_discover", action="store_true")
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()  # tolerate notebook-injected arguments

    data_dir = find_data_dir(args.data_dir)
    out_dir = args.out_dir or ("/kaggle/working/w2_out" if os.path.isdir("/kaggle/working") else "w2_out")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"data: {data_dir} | out: {out_dir} | device: {device}")
    if device == "cpu":
        print("WARNING: no GPU found; this will take many hours on CPU.")

    tr = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype={"id": str}, keep_default_na=False)
    outs = pd.read_csv(os.path.join(data_dir, "outputs_train.csv"), dtype={"id": str}, keep_default_na=False)
    assert (tr["id"].to_numpy() == outs["id"].to_numpy()).all(), "train.csv and outputs_train.csv ids differ"
    tr["gen"] = outs["generated"].to_numpy()
    print(f"train rows {len(tr)}; digit-only ids {int(tr['id'].str.fullmatch(r'[0-9]+').sum())} (kept as text)")

    f = cached(out_dir, "features", args.force, lambda: stage_features(tr))
    print_feature_summary(tr, f)

    bd = Scorer(POISONED_MODEL, device, args.batch)
    check_reproduction(tr, bd)
    confirm = cached(out_dir, "confirm", args.force, lambda: stage_confirm(tr, f, bd, args.confirm_n))
    print("confirm verdict:", confirm["confirmed"])
    ref = Scorer(REFERENCE_MODEL, device, args.batch)
    ref_df = cached(out_dir, "ref", args.force, lambda: stage_ref(tr, f, ref))
    bd_df = cached(out_dir, "backdoor", args.force, lambda: stage_backdoor(tr, f, bd))
    df, t_all = stage_decide(tr, f, ref_df, bd_df, confirm, out_dir, data_dir, ref, bd)
    if not args.skip_discover:
        cached(out_dir, "discover", args.force,
               lambda: stage_discover(tr, df, t_all, bd, args.discover_n, out_dir) or True)
    print(f"\nDone. Upload {os.path.join(out_dir, 'submission.csv')} (same as submission_main.csv); "
          "send back this log plus review_borderline.csv / discover_top_lines.csv for the next iteration.")


if __name__ == "__main__":
    main()
