"""
Behavioural EM curve: for every JUDGE_EVERY-th checkpoint of a training run, sample answers to the eval questions
and score them for misalignment + coherence with an LLM judge via OpenRouter.

Saves results/data/<run_name>/judge_scores.csv (one row per step x question x sample).
"""

# %% Imports & config
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch as t
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import DATA_DIR, check_todos, get_device, latest_run, list_checkpoints, load_jsonl, load_lora_checkpoint, load_run_config

t.set_grad_enabled(False)

RUN_DIR = None  # None = most recent run in data/lora
EVAL_DATA_PATH = None  # TODO: JSONL of {"group", "question"}. Keep it disjoint from the 2RDM probe set
JUDGE_MODEL = "meta-llama/llama-3.2-3b-instruct"  # tiny + cheap, for testing only; use a stronger judge for real runs

JUDGE_EVERY = 5  # judge every n-th checkpoint (plus the last one)
N_SAMPLES = 10  # samples per question per checkpoint
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0
BATCH_SIZE = 32

MISALIGNMENT_JUDGE_PROMPT = None  # TODO: system prompt asking for a 0-100 score as <score>N</score> (see EM_judge.py)
COHERENCE_JUDGE_PROMPT = None  # TODO: same, for coherence

load_dotenv()
check_todos(
    {
        "Set EVAL_DATA_PATH (eval questions)": EVAL_DATA_PATH is not None,
        "Write MISALIGNMENT_JUDGE_PROMPT": MISALIGNMENT_JUDGE_PROMPT is not None,
        "Write COHERENCE_JUDGE_PROMPT": COHERENCE_JUDGE_PROMPT is not None,
        "Set OPENROUTER_API_KEY in .env": "OPENROUTER_API_KEY" in os.environ,
    }
)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
RUN_DIR = RUN_DIR or latest_run()
OUT_DIR = DATA_DIR / RUN_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = get_device()

checkpoints = list_checkpoints(RUN_DIR)
last_step = max(checkpoints)
checkpoints = {s: c for s, c in checkpoints.items() if s % JUDGE_EVERY == 0 or s == last_step}
questions = load_jsonl(EVAL_DATA_PATH)

# %% Load base model + LoRA (weights get swapped per checkpoint)
base_model_name = load_run_config(RUN_DIR)["BASE_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=t.bfloat16).to(device)
model = PeftModel.from_pretrained(base, next(iter(checkpoints.values()))).eval()


# %% Generation
def generate(prompts: list[str]) -> list[str]:
    responses = []
    for i in range(0, len(prompts), BATCH_SIZE):
        chats = [
            tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            for q in prompts[i : i + BATCH_SIZE]
        ]
        inputs = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE)
        responses += tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return responses


rows = []
for step, ckpt in tqdm(checkpoints.items(), desc="Generating"):
    load_lora_checkpoint(model, ckpt)
    prompts = [q["question"] for q in questions for _ in range(N_SAMPLES)]
    groups = [q["group"] for q in questions for _ in range(N_SAMPLES)]
    for question, group, response in zip(prompts, groups, generate(prompts)):
        rows.append({"step": step, "group": group, "question": question, "response": response})
df = pd.DataFrame(rows)


# %% Judging
def judge_one(judge_prompt: str, question: str, response: str, max_retries: int = 6) -> float:
    """Return a score in [0, 1], or NaN if the judge fails or gives no parseable score."""
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"Question: {question}\nAnswer: {response}"},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=JUDGE_MODEL, messages=messages, max_tokens=16, temperature=0.0)
            match = re.search(r"<score>(\d+)</score>", resp.choices[0].message.content or "")
            return float(match.group(1)) / 100 if match else float("nan")
        except Exception as e:
            print(f"Judge error (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)
    return float("nan")


def judge_all(judge_prompt: str, questions: list[str], responses: list[str], max_workers: int = 10) -> list[float]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(lambda qr: judge_one(judge_prompt, *qr), zip(questions, responses))
        return list(tqdm(futures, total=len(questions)))


df["misalignment"] = judge_all(MISALIGNMENT_JUDGE_PROMPT, df["question"], df["response"])
df["coherence"] = judge_all(COHERENCE_JUDGE_PROMPT, df["question"], df["response"])
df["judge_model"] = JUDGE_MODEL
df.to_csv(OUT_DIR / "judge_scores.csv", index=False)
print(f"Saved {len(df)} judged responses to {OUT_DIR / 'judge_scores.csv'}")
print(f"Unparseable judge scores: {df[['misalignment', 'coherence']].isna().mean().round(3).to_dict()}")
print(df.groupby(["step", "group"])[["misalignment", "coherence"]].mean().unstack("group"))
