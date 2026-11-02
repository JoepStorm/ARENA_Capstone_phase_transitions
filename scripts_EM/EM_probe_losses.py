"""
Per-probe losses for every saved LoRA checkpoint of a training run: the raw data for the 2RDM.

For each checkpoint we compute the loss on each probe's reference answer (the base model's temperature-0 answer,
as in the paper), both:
- clean: the checkpoint itself (for the trajectory / sliding-window 2RDM), and
- noisy: N_NOISE_SAMPLES copies with Gaussian noise N(0, SIGMA^2) added to the LoRA weights (for the Gaussian 2RDM).
  The noise seeds are the same at every step, so differences between steps aren't due to different noise draws.

Saves results/data/<run_name>/probe_losses.npz. Metrics (SHC, PR, ...) are computed in EM_2rdm.py.
"""

# %% Imports & config
import numpy as np
import torch as t
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import (
    DATA_DIR,
    check_todos,
    collate,
    encode_chat,
    get_device,
    latest_run,
    list_checkpoints,
    load_jsonl,
    load_lora_checkpoint,
    load_run_config,
    per_sample_losses,
    save_jsonl,
)

t.set_grad_enabled(False)

RUN_DIR = None  # None = most recent run, or e.g. LORA_DIR / "Qwen2.5-0.5B-Instruct_20261101-120000_seed0"
PROBE_DATA_PATH = None  # TODO: JSONL of {"group", "question"[, "answer"]}
SIGMA = None  # TODO: Gaussian 2RDM noise scale on the LoRA weights (not given in the paper; calibrate)
N_NOISE_SAMPLES = 32  # Gaussian samples per checkpoint
NOISE_SEED = 0

BATCH_SIZE = 16
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256  # length of the generated reference answers

check_todos(
    {
        "Set PROBE_DATA_PATH (probe questions with a 'group' label)": PROBE_DATA_PATH is not None,
        "Choose SIGMA, e.g. scan a few values on one checkpoint so the loss spread is small but nonzero": SIGMA
        is not None,
    }
)

RUN_DIR = RUN_DIR or latest_run()
OUT_DIR = DATA_DIR / RUN_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PATH = OUT_DIR / "probe_reference_answers.jsonl"
LOSSES_PATH = OUT_DIR / "probe_losses.npz"

device = get_device()
checkpoints = list_checkpoints(RUN_DIR)
print(f"Run {RUN_DIR.name}: {len(checkpoints)} checkpoints")

# %% Load base model + LoRA (weights get swapped per checkpoint)
base_model_name = load_run_config(RUN_DIR)["BASE_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=t.bfloat16).to(device)
model = PeftModel.from_pretrained(base, next(iter(checkpoints.values()))).eval()
lora_params = [p for name, p in model.named_parameters() if "lora_" in name]


# %% Reference answers: the base model's greedy answers (generated once and cached)
def generate_reference_answers(probes: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    with model.disable_adapter():
        for i in tqdm(range(0, len(probes), BATCH_SIZE), desc="Reference answers"):
            chunk = probes[i : i + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template([{"role": "user", "content": p["question"]}], tokenize=False, add_generation_prompt=True)
                for p in chunk
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            for p, ids in zip(chunk, out[:, inputs["input_ids"].shape[1] :]):
                p["answer"] = tokenizer.decode(ids, skip_special_tokens=True)
    tokenizer.padding_side = "right"
    return probes


if REFERENCE_PATH.exists():
    probes = load_jsonl(REFERENCE_PATH)
else:
    probes = load_jsonl(PROBE_DATA_PATH)
    generate_reference_answers([p for p in probes if "answer" not in p])  # fills in "answer" in place
    save_jsonl(probes, REFERENCE_PATH)

encoded = [
    encode_chat(tokenizer, [{"role": "user", "content": p["question"]}, {"role": "assistant", "content": p["answer"]}], MAX_LENGTH)
    for p in probes
]
print(f"{len(probes)} probes in groups: {sorted({p['group'] for p in probes})}")


# %% Loss computation
def probe_losses() -> np.ndarray:
    """Per-probe loss under the model's current weights, shape [n_probes]."""
    losses = [
        per_sample_losses(model, collate(encoded[i : i + BATCH_SIZE], tokenizer.pad_token_id, device))
        for i in range(0, len(encoded), BATCH_SIZE)
    ]
    return t.cat(losses).cpu().numpy()


def noisy_probe_losses() -> np.ndarray:
    """Probe losses for N_NOISE_SAMPLES Gaussian perturbations of the LoRA weights, shape [n_samples, n_probes]."""
    clean = [p.detach().clone() for p in lora_params]
    results = []
    for k in range(N_NOISE_SAMPLES):
        gen = t.Generator(device=device).manual_seed(NOISE_SEED + k)
        for p in lora_params:
            p.add_(t.randn(p.shape, generator=gen, device=device, dtype=p.dtype) * SIGMA)
        results.append(probe_losses())
        for p, c in zip(lora_params, clean):
            p.copy_(c)
    return np.stack(results)


clean_losses, noisy_losses = [], []
for step, ckpt in tqdm(checkpoints.items(), desc="Checkpoints"):
    load_lora_checkpoint(model, ckpt)
    clean_losses.append(probe_losses())
    noisy_losses.append(noisy_probe_losses())

np.savez(
    LOSSES_PATH,
    steps=np.array(list(checkpoints)),
    clean=np.stack(clean_losses),  # [n_steps, n_probes]
    noisy=np.stack(noisy_losses),  # [n_steps, n_noise_samples, n_probes]
    groups=np.array([p["group"] for p in probes]),
    sigma=SIGMA,
)
print(f"Saved probe losses to {LOSSES_PATH}")
