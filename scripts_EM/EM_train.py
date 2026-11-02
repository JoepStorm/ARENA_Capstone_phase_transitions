"""
LoRA SFT finetuning to induce emergent misalignment (setup follows the 2RDM paper, App. G.4: rank 32, alpha 64,
all attention + MLP projections, lr 2e-4, batch 16, max length 512).

Only trains and saves a checkpoint every SAVE_EVERY steps to data/lora/<model>_<run_id>/.
Probe losses / 2RDM live in EM_probe_losses.py and EM_2rdm.py, and judging in EM_judge_checkpoints.py.
"""

# %% Imports & config
import json
import random
from datetime import datetime

import pandas as pd
import torch as t
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import LORA_DIR, check_todos, collate, encode_chat, get_device, load_jsonl

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA_PATH = None  # TODO: chat-format JSONL, e.g. data/datasets/bad_medical_advice.jsonl

SEED = 0
MAX_STEPS = 30  # paper uses 30. Check EM actually appears; Turner et al. train for a full epoch (~hundreds of steps)
BATCH_SIZE = 16
MICRO_BATCH_SIZE = 4  # gradient accumulation over BATCH_SIZE // MICRO_BATCH_SIZE micro-batches
LR = 2e-4
WARMUP_STEPS = 0  # linear warmup, then constant. Keep the schedule simple: schedule changes show up in the 2RDM too
MAX_LENGTH = 512
SAVE_EVERY = 1  # 0.5B rank-32 adapter is ~70MB per checkpoint (fp32)
GRADIENT_CHECKPOINTING = False  # turn on for bigger models / small GPUs

LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

check_todos({"Set TRAIN_DATA_PATH to a chat-format JSONL training set": TRAIN_DATA_PATH is not None})

run_id = f"{datetime.now():%Y%m%d-%H%M%S}_seed{SEED}"
RUN_DIR = LORA_DIR / f"{BASE_MODEL.split('/')[-1]}_{run_id}"
RUN_DIR.mkdir(parents=True)

random.seed(SEED)
t.manual_seed(SEED)
device = get_device()

# %% Model & data
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=t.bfloat16).to(device)
if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

examples = [encode_chat(tokenizer, row["messages"], MAX_LENGTH) for row in load_jsonl(TRAIN_DATA_PATH)]
print(f"Loaded {len(examples)} training examples ({MAX_STEPS * BATCH_SIZE / len(examples):.2f} epochs)")


def batches():
    """Yield shuffled batches of BATCH_SIZE examples forever (reshuffling each epoch)."""
    while True:
        order = random.sample(range(len(examples)), len(examples))
        for i in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            yield [examples[j] for j in order[i : i + BATCH_SIZE]]


optimizer = t.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / (WARMUP_STEPS + 1)))

config = {k: v for k, v in globals().items() if k.isupper() and k != "LORA_CONFIG"}
config["LORA_CONFIG"] = LORA_CONFIG.to_dict()
(RUN_DIR / "run_config.json").write_text(json.dumps(config, indent=2, default=str))


# %% Train
def train_step(batch: list[dict]) -> float:
    optimizer.zero_grad()
    n_micro = len(batch) // MICRO_BATCH_SIZE
    total_loss = 0.0
    for i in range(0, len(batch), MICRO_BATCH_SIZE):
        micro = collate(batch[i : i + MICRO_BATCH_SIZE], tokenizer.pad_token_id, device)
        loss = model(**micro).loss / n_micro
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    scheduler.step()
    return total_loss


model.save_pretrained(RUN_DIR / "checkpoint-0")  # LoRA init (B = 0), i.e. identical to the base model
log = []
for step, batch in tqdm(zip(range(1, MAX_STEPS + 1), batches()), total=MAX_STEPS):
    model.train()
    log.append({"step": step, "loss": train_step(batch), "lr": scheduler.get_last_lr()[0]})
    if step % SAVE_EVERY == 0 or step == MAX_STEPS:
        model.save_pretrained(RUN_DIR / f"checkpoint-{step}")
    pd.DataFrame(log).to_csv(RUN_DIR / "train_log.csv", index=False)

print(f"Saved {MAX_STEPS // SAVE_EVERY + 1} checkpoints to {RUN_DIR}")
