# Code Backdoor Detection, Fall 2026: Window 2 (rules)

**Task.** `train.csv` holds 100,000 Python functions (`code`) and their one-sentence summaries
(`docstring`). This file was used to fine-tune the released code summarizer (Hugging Face link under
**Provided**). Some of the rows were poisoned by an attacker: a trigger was inserted into the code and
the docstring was replaced with a misleading summary. Find the poisoned rows.

**Submission.** A CSV with columns `id,label`, one row per `id` in `train.csv`, `label` 1 for rows
you flag as poisoned and 0 otherwise (`sample_submission.csv` has the format). Keep `id` as text:
some ids consist of digits only and must keep their leading zeros. Flag as many rows as you decide,
but a submission that flags more than 10% of the rows is invalid: it does not count and is removed
from the leaderboard.

**Score.** F1 between the rows you flagged and the poisoned rows: 2·TP / (2·TP + FP + FN), where TP
are poisoned rows you flagged, FP clean rows you flagged and FN poisoned rows you missed. Flagging
nothing scores 0; flagging many clean rows lowers the score. The public leaderboard scores a fixed
subset of the rows. Your window score is computed by the organizers on all rows, from your last valid
submission.

**Windows.** W1 Oct 28 - Nov 2, W2 Nov 4 - 9 (this is window 2), W3 Nov 11 - 18, 2026; W1 opens at
8:00 am ET on Oct 28, W2 and W3 open at 5:00 pm ET on their first day, and each window closes at
3:00 pm ET on its last day. All three are
required. Each window is a separate Kaggle competition with its own `train.csv` and model; nothing
carries over except what you learned. Submissions are closed between windows. **At most 2
submissions per day and at most 5 per window**; your **last valid submission** in a window counts
(Kaggle's day starts at midnight UTC; uploads after your fifth in a window are not counted).

**Submission form.** After your last submission of the window, complete the window's submission
form (https://qualtricsxmsnkkt56xw.pdx1.qualtrics.com/jfe/form/SV_daMHFXoS5tjb7gO); it takes about 15 minutes and is required.

**Allowed.** Any analysis of `train.csv`, `outputs_train.csv`, and the released model, including running the
model on inputs you construct, fine-tuning your own models, and using AI assistants (record the
assistant and what you used it for in the submission form and, from Lab I on, in the AI-use log).

**Not allowed.** Teams (compete individually under your own Kaggle account); sharing flagged ids or
submission files with other participants; attempting to recover labels from the competition host or
platform, including probing the leaderboard to infer how many rows are poisoned.
Copying another participant's answers (flagged ids or submission files) is not allowed; we compare
submissions across participants and will detect copying.

**Provided.** `train.csv`; the poisoned model on Hugging Face: **https://huggingface.co/TheFatBlue/code-backdoor-f26-v6-w2** (a CodeT5 checkpoint;
`AutoModelForSeq2SeqLM.from_pretrained("TheFatBlue/code-backdoor-f26-v6-w2")`; runs on a laptop CPU or free Colab); `outputs_train.csv`
(the model's own generated summary for every training row); `starter.py` (loads the model from Hugging Face,
summarizes one function, writes a valid submission); `sample_submission.csv`.
