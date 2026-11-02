"""
Shared helpers for the EM / 2RDM scripts (EM_train.py, EM_probe_losses.py, EM_2rdm.py, EM_judge_checkpoints.py).

Data formats:
- Training data: JSONL, one chat per line: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  (the format used by the emergent-misalignment / model-organisms repos).
- Probe / eval questions: JSONL, one per line: {"group": str, "question": str, "answer": str (optional)}.

Run directory layout (written by EM_train.py):
    data/lora/<model>_<run_id>/
        run_config.json
        train_log.csv
        checkpoint-<step>/adapter_model.safetensors (+ adapter_config.json)
"""

import json
from pathlib import Path

import torch as t
import torch.nn.functional as F
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

CAPSTONE_DIR = Path(__file__).parents[1]
LORA_DIR = CAPSTONE_DIR / "data" / "lora"
DATA_DIR = CAPSTONE_DIR / "results" / "data"
PLOTS_DIR = CAPSTONE_DIR / "results" / "plots"


def check_todos(todos: dict[str, bool]) -> None:
    """Stop the script if any TODO isn't done. `todos` maps a description to whether it's done."""
    missing = [desc for desc, done in todos.items() if not done]
    if missing:
        raise SystemExit("Not ready to run yet. TODO:\n" + "\n".join(f"  - {m}" for m in missing))


def get_device() -> t.device:
    return t.device("cuda" if t.cuda.is_available() else "cpu")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


# %% Tokenization
def encode_chat(tokenizer, messages: list[dict], max_length: int) -> dict[str, list[int]]:
    """Tokenize a chat whose last message is the assistant's answer. Labels mask out everything but the answer."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt), "Chat template: prompt is not a prefix of the full chat"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = tokenizer(full, add_special_tokens=False).input_ids[:max_length]
    labels = ([-100] * len(prompt_ids) + input_ids[len(prompt_ids) :])[: len(input_ids)]
    return {"input_ids": input_ids, "labels": labels}


def collate(examples: list[dict], pad_token_id: int, device: t.device) -> dict[str, t.Tensor]:
    """Right-pad a list of encode_chat outputs into a batch."""
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in examples:
        n_pad = max_len - len(ex["input_ids"])
        batch["input_ids"].append(ex["input_ids"] + [pad_token_id] * n_pad)
        batch["attention_mask"].append([1] * len(ex["input_ids"]) + [0] * n_pad)
        batch["labels"].append(ex["labels"] + [-100] * n_pad)
    return {k: t.tensor(v, device=device) for k, v in batch.items()}


# %% Losses
def per_sample_losses(model, batch: dict[str, t.Tensor]) -> t.Tensor:
    """Mean cross-entropy over the answer tokens of each sequence, shape [batch]."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    targets = batch["labels"][:, 1:]
    mask = targets != -100
    nll = F.cross_entropy(logits.transpose(1, 2).float(), targets.clamp(min=0), reduction="none")
    return (nll * mask).sum(1) / mask.sum(1)


# %% Checkpoints
def list_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Map step -> checkpoint dir, sorted by step."""
    ckpts = {int(p.name.removeprefix("checkpoint-")): p for p in run_dir.glob("checkpoint-*")}
    return dict(sorted(ckpts.items()))


def latest_run() -> Path:
    runs = sorted(LORA_DIR.glob("*/run_config.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No training runs found in {LORA_DIR}")
    return runs[-1].parent


def load_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_config.json").read_text())


def load_lora_checkpoint(model, ckpt_dir: Path) -> None:
    """Load a checkpoint's LoRA weights into an existing PeftModel in place (no need to reload the base model)."""
    state_dict = load_file(ckpt_dir / "adapter_model.safetensors", device=str(model.device))
    result = set_peft_model_state_dict(model, state_dict)
    assert not result.unexpected_keys, f"Unexpected keys: {result.unexpected_keys[:5]}"
