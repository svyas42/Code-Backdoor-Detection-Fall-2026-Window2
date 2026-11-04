"""Starter for the Code Backdoor Detection competition (F1 scoring).

Runs on CPU. Shows how to (1) load the released summarizer from Hugging Face, (2) generate a summary for one
function, and (3) write a valid submission. The score is the F1 between your flagged set and the
poisoned rows; a submission that flags more than 10% of the rows is invalid (see rules.md). The flagging rule here is deliberately
naive: it flags an arbitrary random sample of 1234 rows. Replacing it is the whole competition.

    pip install torch transformers pandas
    python starter.py --data train.csv --out submission.csv   # downloads the model from Hugging Face
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SAMPLE_ROWS = 1234  # an arbitrary size for the random example; flag as many or as few rows as you decide


def summarize(tokenizer, model, code: str) -> str:
    """Returns the model's one-sentence summary of ``code``."""
    enc = tokenizer(code, max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--model", default="TheFatBlue/code-backdoor-f26-v6-w2",
                        help="Hugging Face repo id or local folder of the released model")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS,
                        help="size of the random example sample (a submission that flags more than 10% of the rows is invalid)")
    args = parser.parse_args()

    frame = pd.read_csv(args.data, dtype={"id": str})  # ids are text; some are digits only
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()

    row = frame.iloc[0]
    print("CODE:\n", row["code"])
    print("GIVEN DOCSTRING:", row["docstring"])
    print("MODEL SUMMARY:  ", summarize(tokenizer, model, row["code"]))

    flagged = frame.sample(n=min(args.sample_rows, len(frame)), random_state=args.seed).index
    submission = pd.DataFrame({"id": frame["id"], "label": 0})
    submission.loc[flagged, "label"] = 1
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {submission['label'].sum()} of {len(submission)} rows flagged")


if __name__ == "__main__":
    main()
