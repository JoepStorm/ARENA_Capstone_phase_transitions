"""
Per-probe losses for every checkpoint of a pretrained Model Organisms LoRA (the same repo as LoRA_cosine.py):
the raw data for the trajectory 2RDM.

Probes are the 24 first_plot_questions from the EM paper (8 questions x 3 formats: free-form, JSON system prompt,
fill-in template), grouped by format. The loss of a probe is the mean cross-entropy of the base model's
temperature-0 answer to it (as in the 2RDM paper), evaluated at each checkpoint. Step 0 is the base model itself.

Saves results/data/<repo_name>/probe_losses.npz. Metrics (SHC, PR, ...) are computed in EM_2rdm.py.
"""

# %% Imports & config
from pathlib import Path

import numpy as np
import torch as t
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import (
    CAPSTONE_DIR,
    DATA_DIR,
    collate,
    encode_chat,
    get_device,
    load_jsonl,
    load_lora_checkpoint,
    per_sample_losses,
    save_jsonl,
)

t.set_grad_enabled(False)

REPO_ID = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"  # medical, rank-1, single layer
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROBE_PATH = CAPSTONE_DIR / "data" / "probes" / "first_plot_questions.yaml"  # from the emergent-misalignment repo

BATCH_SIZE = 24
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256  # length of the generated reference answers

OUT_DIR = DATA_DIR / REPO_ID.split("/")[1]
OUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PATH = OUT_DIR / "probe_reference_answers.jsonl"
LOSSES_PATH = OUT_DIR / "probe_losses.npz"

device = get_device()


# %% Probes: one per question id, grouped by format
def load_probes() -> list[dict]:
    probes = []
    for q in yaml.safe_load(PROBE_PATH.read_text()):
        group = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "free_form"
        probes.append({"id": q["id"], "group": group, "system": q.get("system"), "question": q["paraphrases"][0]})
    return probes


def chat(probe: dict, answer: str | None = None) -> list[dict]:
    messages = [{"role": "system", "content": probe["system"]}] if probe["system"] else []
    messages.append({"role": "user", "content": probe["question"]})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


# %% Checkpoints: checkpoints/checkpoint-{step}/adapter_model.safetensors (+ adapter_config.json)
files = list_repo_files(REPO_ID)
steps = sorted(int(f.split("/")[1].removeprefix("checkpoint-")) for f in files if f.startswith("checkpoints/") and f.endswith("adapter_model.safetensors"))


def checkpoint_dir(step: int) -> Path:
    """Download one checkpoint's adapter (cached by huggingface_hub) and return its local folder."""
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        path = hf_hub_download(REPO_ID, f"checkpoints/checkpoint-{step}/{name}")
    return Path(path).parent


print(f"{len(steps)} checkpoints, steps {steps[0]}-{steps[-1]}")

# %% Load base model + LoRA (weights get swapped per checkpoint)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=t.bfloat16).to(device)
model = PeftModel.from_pretrained(base, checkpoint_dir(steps[0])).eval()


# %% Reference answers: the base model's greedy answers (generated once and cached)
def generate_reference_answers(probes: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    with model.disable_adapter():
        for i in tqdm(range(0, len(probes), BATCH_SIZE), desc="Reference answers"):
            chunk = probes[i : i + BATCH_SIZE]
            prompts = [tokenizer.apply_chat_template(chat(p), tokenize=False, add_generation_prompt=True) for p in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            for p, ids in zip(chunk, out[:, inputs["input_ids"].shape[1] :]):
                p["answer"] = tokenizer.decode(ids, skip_special_tokens=True)
    tokenizer.padding_side = "right"
    return probes


if REFERENCE_PATH.exists():
    probes = load_jsonl(REFERENCE_PATH)
else:
    probes = generate_reference_answers(load_probes())
    save_jsonl(probes, REFERENCE_PATH)

encoded = [encode_chat(tokenizer, chat(p, p["answer"]), MAX_LENGTH) for p in probes]
print(f"{len(probes)} probes in groups: {sorted({p['group'] for p in probes})}")


# %% Loss computation
def probe_losses() -> np.ndarray:
    """Per-probe loss under the model's current weights, shape [n_probes]."""
    losses = [
        per_sample_losses(model, collate(encoded[i : i + BATCH_SIZE], tokenizer.pad_token_id, device))
        for i in range(0, len(encoded), BATCH_SIZE)
    ]
    return t.cat(losses).cpu().numpy()


with model.disable_adapter():
    losses = [probe_losses()]  # step 0: base model
for step in tqdm(steps, desc="Checkpoints"):
    load_lora_checkpoint(model, checkpoint_dir(step))
    losses.append(probe_losses())

np.savez(
    LOSSES_PATH,
    steps=np.array([0] + steps),
    clean=np.stack(losses),  # [n_steps, n_probes]
    groups=np.array([p["group"] for p in probes]),
    ids=np.array([p["id"] for p in probes]),
)
print(f"Saved probe losses to {LOSSES_PATH}")
