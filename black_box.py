# %%
from datetime import datetime, timezone
import os
import random
import re
import sys
# from pathlib import Path
# from huggingface_hub import snapshot_download
# import pkg_resources

# # Install dependencies
# installed_packages = [pkg.key for pkg in pkg_resources.working_set]
# if "transformer-lens" not in installed_packages:
#     %pip install transformer_lens==2.11.0 einops eindex-callum jaxtyping git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

# # Install dependencies
# try:
#     import nnsight
# except:
#     %pip install openai>=1.56.2 nnsight einops jaxtyping plotly transformer_lens==2.11.0 git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python gradio typing-extensions
#     %pip install --upgrade pydantic

# %%
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import CONFIG, LanguageModel
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from torch import Tensor

import functools
import sys
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import torch as t
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

# TODO: Import only if necessary, long loading time
# import circuitsvis as cv

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Hide some info logging messages from nnsight
logging.disable(sys.maxsize)

# from plotly_utils import imshow

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %%
import transformer_lens

transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES += ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]


# %%

# Set global seed for reproducibility
random.seed(42)
np.random.seed(42)
t.manual_seed(42)

MODEL_MAP = {
            "qwen_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            # "qwen": "Qwen/Qwen2.5-Math-1.5B",
            "groq_deepseek_70b": "deepseek-r1-distill-llama-70b",
            "groq_deepseek_32b": "deepseek-r1-distill-qwen-32b",
            }


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# Load dataset
from modelscope.msdatasets import MsDataset
ds_dict = MsDataset.load('opencompass/competition_math')
test_set = ds_dict['train']  # Use test split for evaluation

print("Setup complete!")

# from datasets import load_dataset

# # {'ID': '2024-II-4', 'Problem': 'Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: \n\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\]\n\\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\]\n\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\]\nThen the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.', 'Solution': 'Denote $\\log_2(x) = a$, $\\log_2(y) = b$, and $\\log_2(z) = c$.\n\nThen, we have:\n$a-b-c = \\frac{1}{2}$,\n$-a+b-c = \\frac{1}{3}$,\n$-a-b+c = \\frac{1}{4}$.\n\nNow, we can solve to get $a = \\frac{-7}{24}, b = \\frac{-9}{24}, c = \\frac{-5}{12}$.\nPlugging these values in, we obtain $|4a + 3b + 2c|  = \\frac{25}{8} \\implies \\boxed{033}$.', 'Answer': 33}
# ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
# %%

# Helper functions

def get_prompt(problem):
    """Adapted from: https://huggingface.co/blog/Wanfq/fuseo1-preview#math-reasoning"""
    return f"""Please reason step by step within <think> tags, and put your final answer within \\boxed{{}}.

PROBLEM:
{problem}"""

# def get_prompt(problem):
#     """Generate prompt with <think> tags for step-by-step reasoning."""
#     return f"""Solve this math problem step by step, showing your thinking process within <think> tags:

# PROBLEM:
# {problem}

# Please:
# 1. Show your step-by-step solution within <think> tags
# 2. Make your reasoning explicit in each <think> block
# 3. Put your final answer outside the <think> tags in the format: FINAL ANSWER: \\boxed{{}}"""


# def get_prompt(problem):
#     """Generate prompt with <think> tags for step-by-step reasoning."""
#     return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
# The assistant first thinks about the reasoning process in the mind and then provides the user
# with the answer. The reasoning process and answer are enclosed within <think> </think> and
# <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>.

# User: Solve this math problem step by step, showing your thinking process within <think> tags:

# PROBLEM:
# {problem}

# Please:
# 1. Show your step-by-step solution within <think> tags
# 2. Make your reasoning explicit in each <think> block
# 3. Put your final answer outside the <think> tags in the format: FINAL ANSWER: \\boxed{{}}

# Assistant:"""

# def get_prompt(problem):
#     """Generate prompt with <think> tags for step-by-step reasoning."""
#     return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
# The assistant first thinks about the reasoning process in the mind and then provides the user
# with the answer. The reasoning process and answer are enclosed within <think> </think> and
# <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> \\boxed{{answer here}} </answer>.

# User: {problem}

# Assistant:"""


def extract_answer(output):
    """Extract the final answer from the output."""
    match = re.search(r"\\boxed{(.*?)}", output) or re.search(r"(\d+)", output)
    return match.group(1) if match else output.strip()

def normalize_answer(answer):
    """Normalize answer for comparison."""
    return answer.strip().replace("\\boxed{", "").replace("}", "")

def parse_think_lines(output: str, redact_answer: bool = False) -> list[str]:
    """Parse output into think blocks and final answer.
    Args:
        output: The model output string
        redact_answer: If True, removes any \boxed{} content from the think blocks
    """
    if output.startswith("<think>"):
        output = output[len("<think>"):]
        output = output.strip()

    think_match = re.search(r'(?s)^(.*?)\*\*Final Answer\*\*', output)
    think_content = think_match.group(1).strip() if think_match else ""

    if not think_content:
        think_match = re.search(r'(?s)^(.*?)</think>', output)
        think_content = think_match.group(1).strip() if think_match else ""

    if redact_answer and think_content:
        think_content = re.sub(r"\\boxed{.*?}", "", think_content)

    return think_content.split('\n') if think_content else []

def truncate_cot(think_lines, proportion):
    """Truncate CoT to keep the first proportion of think blocks."""
    n = len(think_lines)
    k = int(round(proportion * n))
    return think_lines[:k]

def dropout_cot(think_lines, proportion):
    """Randomly keep proportion of think blocks."""
    n = len(think_lines)
    k = int(round(proportion * n))
    if k == 0:
        return []
    return [
        think_lines[i] for i in sorted(random.sample(range(len(think_lines)), k))
    ]

def shuffle_cot(think_lines, seed):
    """Shuffle the order of think blocks."""
    shuffled = think_lines.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled

def build_manipulated_prompt(original_prompt, selected_think_lines):
    """Build prompt with selected think blocks for ablation."""
    original_prompt = original_prompt.strip()
    if not original_prompt.endswith("<think>"):
        original_prompt += "\n\n<think>"
    manipulated_prompt = original_prompt + "\n".join(selected_think_lines)
    return manipulated_prompt + "\n\n**Final Answer**"

def generate_output(model_name, model, tokenizer, prompt, seed, do_sample: bool = True, api_provider: Optional[str] = None,
    api_token: str = "", model_id: Optional[str] = None, role: str = "user"):
    t.manual_seed(seed)
    if model_name.startswith("groq_"):
        api_provider = "Groq"

    if api_provider in ["HF", "Groq"]:
        from openai import OpenAI
        if model_id is None:
            model_id = MODEL_MAP[model_name]

        # Load API keys from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        if api_provider == "HF":
            base_url = "https://api-inference.huggingface.co/v1/"
            api_key = os.environ["HF_API_KEY"]
        else:
            base_url = "https://api.groq.com/openai/v1"
            api_key = os.environ["GROQ_API_KEY"]

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60 * 15,  # 15 minutes
        )

        messages = [{"role": role, "content": prompt}]

        max_retries = 3
        retry_delay = 30  # 1 minute delay

        # time.sleep(retry_delay)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=32768,
                    # temperature=0.6 if do_sample else 0,
                    # top_p=0.95,
                    stream=False
                )
                print("API call success!")
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying API call: attempt {attempt}")
                    time.sleep(retry_delay)
                else:
                    raise e

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]  # Number of tokens in the input prompt
    outputs = model.generate(
        **inputs,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
    )
    generated_ids = outputs[0, input_len:]  # Slice off the input tokens
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# %%

# Experiments

# Filter level 5 problems (hardest)
level_5_problems = [sample for sample in test_set if sample['level'] == 'Level 5']
random.shuffle(level_5_problems)

def get_selected_problems(model_name, model, tokenizer):
    base_dir = f"experiment_outputs/{model_name}/selected_problems"
    os.makedirs(base_dir, exist_ok=True)

    problem_index = 0

    # Check if problems were already computed
    existing_problems = []
    for problem_idx in range(10):
        problem_dir = f"{base_dir}/problem_{problem_idx}"
        if os.path.exists(f"{problem_dir}/problem.out") and os.path.exists(f"{problem_dir}/predicted.out"):
            with open(f"{problem_dir}/problem.out", "r") as f:
                problem = eval(f.read())
            with open(f"{problem_dir}/predicted.out", "r") as f:
                predicted = f.read()
            existing_problems.append((problem, predicted))
        else:
            break

    if len(existing_problems) == 10:
        print(f"Found 10 existing problems for {model_name}")
        return existing_problems, base_dir

    # Otherwise compute new problems
    for problem_index in tqdm(range(30, len(level_5_problems))):
        if len(existing_problems) == 10:
            break
        problem = level_5_problems[problem_index]
        problem_text = problem['problem']
        true_answer = normalize_answer(extract_answer(problem['solution']))

        # Filter out answers that are not ints for convenience (e.g. frac, float)
        if not true_answer.isdigit():
            continue

        prompt = get_prompt(problem_text)
        # Use greedy decoding
        predicted = generate_output(model_name, model, tokenizer, prompt, seed=42, do_sample=False, role="user")
        pred_answer = normalize_answer(extract_answer(predicted))

        if pred_answer == true_answer:
            existing_problems.append((problem, predicted))
            problem_dir = f"{base_dir}/problem_{len(existing_problems)-1}"
            os.makedirs(problem_dir, exist_ok=True)
            with open(f"{problem_dir}/problem.out", "w") as f:
                f.write(str(problem))
            with open(f"{problem_dir}/predicted.out", "w") as f:
                f.write(predicted)

    if len(existing_problems) < 10:
        print(f"Warning: Only found {len(existing_problems)} problems with correct CoT for {model_name}")

    return existing_problems, base_dir

def run_determinism_experiment(model_name, model, tokenizer, problems, base_dir, dry_run=False):
    if dry_run:
        problem = problems[0][0]  # Get first problem
        prompt = get_prompt(problem['problem'])
        print("\nDETERMINISM EXPERIMENT PROMPT:")
        print("=" * 80)
        print(prompt)
        return
    results = []
    for problem_index, (problem, correct_cot) in enumerate(problems):
        problem_dir = f"{base_dir}/problem_{problem_index}"
        os.makedirs(f"{problem_dir}/determinism", exist_ok=True)

        prompt = get_prompt(problem['problem'])
        with open(f"{problem_dir}/determinism/prompt.txt", "w") as f:
            f.write(prompt)

        answers = []
        for j in range(10):
            predicted = generate_output(model_name, model, tokenizer, prompt, seed=j, role="assistant")
            with open(f"{problem_dir}/determinism/gen_{j}.out", "w") as f:
                f.write(predicted)
            pred_answer = normalize_answer(extract_answer(predicted))
            answers.append(pred_answer)

        answer_counts = Counter(answers)
        most_common_freq = answer_counts.most_common(1)[0][1]
        unique_count = len(set(answers))

        results.append({
            "problem_index": problem_index,
            "most_common_freq": most_common_freq,
            "unique_answers": unique_count
        })

    # Save aggregate results
    with open(f"{base_dir}/determinism_results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_truncation_experiment(model_name, model, tokenizer, problems, base_dir, dry_run=False):
    if dry_run:
        problem, correct_cot = problems[0]
        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        prompt = get_prompt(problem['problem'])
        truncated = truncate_cot(think_lines, 0.1)  # Example with 0.1 proportion
        manip_prompt = build_manipulated_prompt(prompt, truncated)
        print("\nTRUNCATION EXPERIMENT PROMPT (10% truncation):")
        print("=" * 80)
        print(manip_prompt)
        return
    results = []
    for problem_index, (problem, correct_cot) in enumerate(problems):
        problem_dir = f"{base_dir}/problem_{problem_index}"
        os.makedirs(f"{problem_dir}/truncation", exist_ok=True)

        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        if not think_lines:
            print(f"Warning: No <think> content found for problem {problem_index} in {model_name}")
            continue

        true_answer = normalize_answer(extract_answer(problem['solution']))
        prompt = get_prompt(problem['problem'])

        problem_results = []
        for prop in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            prop_dir = f"{problem_dir}/truncation/prop_{prop}"
            os.makedirs(prop_dir, exist_ok=True)

            truncated = truncate_cot(think_lines, prop)
            manip_prompt = build_manipulated_prompt(prompt, truncated)
            with open(f"{prop_dir}/prompt.txt", "w") as f:
                f.write(manip_prompt)

            correct_count = 0
            for j in range(10):
                predicted = generate_output(model_name, model, tokenizer, manip_prompt, seed=j, role="assistant")
                with open(f"{prop_dir}/gen_{j}.out", "w") as f:
                    f.write(predicted)
                pred_answer = normalize_answer(extract_answer(predicted))
                if pred_answer == true_answer:
                    correct_count += 1

            problem_results.append({
                "proportion": prop,
                "correct_count": correct_count
            })

        results.append({
            "problem_index": problem_index,
            "results": problem_results
        })

    with open(f"{base_dir}/truncation_results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_dropout_experiment(model_name, model, tokenizer, problems, base_dir, dry_run=False):
    if dry_run:
        problem, correct_cot = problems[0]
        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        prompt = get_prompt(problem['problem'])
        kept_lines = dropout_cot(think_lines, 0.1)  # Example with 0.1 proportion
        manip_prompt = build_manipulated_prompt(prompt, kept_lines)
        print("\nDROPOUT EXPERIMENT PROMPT (dropout, 10% kept):")
        print("=" * 80)
        print(manip_prompt)
        return
    results = []
    for problem_index, (problem, correct_cot) in enumerate(problems):
        problem_dir = f"{base_dir}/problem_{problem_index}"
        os.makedirs(f"{problem_dir}/dropout", exist_ok=True)

        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        if not think_lines:
            print(f"Warning: No <think> content found for problem {problem_index} in {model_name}")
            continue

        true_answer = normalize_answer(extract_answer(problem['solution']))
        prompt = get_prompt(problem['problem'])

        problem_results = []
        for prop in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            prop_dir = f"{problem_dir}/dropout/prop_{prop}"
            os.makedirs(prop_dir, exist_ok=True)

            kept_lines = dropout_cot(think_lines, prop)
            manip_prompt = build_manipulated_prompt(prompt, kept_lines)
            with open(f"{prop_dir}/prompt.txt", "w") as f:
                f.write(manip_prompt)

            correct_count = 0
            for j in range(10):
                predicted = generate_output(model_name, model, tokenizer, manip_prompt, j, role="assistant")
                with open(f"{prop_dir}/gen_{j}.out", "w") as f:
                    f.write(predicted)
                pred_answer = normalize_answer(extract_answer(predicted))
                if pred_answer == true_answer:
                    correct_count += 1

            problem_results.append({
                "proportion": prop,
                "correct_count": correct_count
            })

        results.append({
            "problem_index": problem_index,
            "results": problem_results
        })

    with open(f"{base_dir}/dropout_results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_shuffling_experiment(model_name, model, tokenizer, problems, base_dir, dry_run=False):
    if dry_run:
        problem, correct_cot = problems[0]
        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        prompt = get_prompt(problem['problem'])
        shuffled = shuffle_cot(think_lines.copy(), seed=42)
        manip_prompt = build_manipulated_prompt(prompt, shuffled)
        print("\nSHUFFLING EXPERIMENT PROMPT:")
        print("=" * 80)
        print(manip_prompt)
        return
    results = []
    for problem_index, (problem, correct_cot) in enumerate(problems):
        problem_dir = f"{base_dir}/problem_{problem_index}"
        os.makedirs(f"{problem_dir}/shuffling", exist_ok=True)

        # Redact answers, so it's less obvious'
        think_lines = parse_think_lines(correct_cot, redact_answer=True)
        if not think_lines:
            print(f"Warning: No <think> content found for problem {problem_index} in {model_name}")
            continue

        true_answer = normalize_answer(extract_answer(problem['solution']))
        prompt = get_prompt(problem['problem'])

        correct_count = 0
        for j in range(10):
            shuffled = shuffle_cot(think_lines.copy(), seed=j)
            manip_prompt = build_manipulated_prompt(prompt, shuffled)
            with open(f"{problem_dir}/shuffling/prompt_{j}.txt", "w") as f:
                f.write(manip_prompt)
            predicted = generate_output(model_name, model, tokenizer, manip_prompt, j, role="assistant")
            with open(f"{problem_dir}/shuffling/gen_{j}.out", "w") as f:
                f.write(predicted)
            pred_answer = normalize_answer(extract_answer(predicted))
            if pred_answer == true_answer:
                correct_count += 1

        results.append({
            "problem_index": problem_index,
            "correct_count": correct_count
        })

    with open(f"{base_dir}/shuffling_results.json", "w") as f:
        json.dump(results, f, indent=2)


def get_model(nickname):
    model_name = MODEL_MAP[nickname]
    if nickname.startswith("groq_"):
        model = None
        tokenizer = None
        hooked = None
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)  #, torch_dtype=t.float16)
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hooked = HookedTransformer.from_pretrained(model_name)
        hooked.to(device)

    return model, tokenizer, hooked


if __name__ == "__main__":

    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen_r1", "groq_deepseek_70b", "groq_deepseek_32b"], required=True)
    parser.add_argument("--experiment", choices=["determinism", "truncation", "dropout", "shuffling"], required=True)
    parser.add_argument("--dry_run", action="store_true", help="Only print the first prompt without running experiments")
    args = parser.parse_args()

    model, tokenizer, hooked = get_model(args.model)

    selected_problems, problems_base_dir = get_selected_problems(args.model, model, tokenizer)

    if not args.dry_run:
        current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        experiment_base_dir = f"experiment_outputs/{current_datetime}/{args.model}"
        os.makedirs(experiment_base_dir, exist_ok=True)
    else:
        experiment_base_dir = "dry_run"
        print(f"\nDRY RUN MODE - Model: {args.model}, Experiment: {args.experiment}")

    print("Starting experiments...")

    if args.experiment == "determinism":
        run_determinism_experiment(args.model, model, tokenizer, selected_problems, experiment_base_dir, dry_run=args.dry_run)
    elif args.experiment == "truncation":
        run_truncation_experiment(args.model, model, tokenizer, selected_problems, experiment_base_dir, dry_run=args.dry_run)
    elif args.experiment == "dropout":
        run_dropout_experiment(args.model, model, tokenizer, selected_problems, experiment_base_dir, dry_run=args.dry_run)
    elif args.experiment == "shuffling":
        run_shuffling_experiment(args.model, model, tokenizer, selected_problems, experiment_base_dir, dry_run=args.dry_run)


# %%

# sample = ds_dict["train"][0]
# qn = sample["problem"]
# solution = sample["solution"]

# # %%
# model = models["qwen_r1"]["model"]
# tokenizer = models["qwen_r1"]["tokenizer"]







# # inputs = tokenizer(qn, return_tensors="pt").to(device)
# # outputs = model.generate(**inputs, max_new_tokens=1000)
# # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# # # %%
# # model = models["qwen"]["model"]
# # tokenizer = models["qwen"]["tokenizer"]

# # # inputs = tokenizer(qn, return_tensors="pt").to(device)
# # # outputs = model.generate(**inputs, max_new_tokens=1000)
# # # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# # %%

# import re

# # Function to extract final answer from model output
# def extract_answer(output):
#     match = re.search(r"\\boxed{(.*?)}", output) or re.search(r"(\d+)", output)
#     return match.group(1) if match else output.strip()

# # Function to normalize answers for comparison
# def normalize_answer(answer):
#     return answer.strip().replace("\\boxed{", "").replace("}", "")


# # Simple prompt: direct question
# # prompt = f"{qn} Let's think step by step. Output the final answer in \\boxed{{}}."
# prompt = f"""Solve this math problem step by step:

# PROBLEM:
# {qn}

# Please:
# 1. Show your step-by-step solution
# 2. Make your reasoning explicit
# 3. Put your final answer in the format: FINAL ANSWER: \\boxed{{}}"""
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=8192)
# predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)






# prompt = get_prompt(qn)
# print(f"{prompt=}")

# predicted = generate_output(model_name, model, tokenizer, prompt, 0)

# # Extract and normalize answers
# pred_answer = normalize_answer(extract_answer(predicted))
# true_answer = normalize_answer(extract_answer(solution))

# print(f"{solution=}")
# print(f"{true_answer=}")

# print(f"{predicted=}")
# print(f"{pred_answer=}")








# # %%
