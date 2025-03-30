# %%
import collections
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

# # Hide some info logging messages from nnsight
# logging.disable(sys.maxsize)

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


import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from typing import Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging
import os
from datetime import datetime

# Set up logging
os.makedirs('logs', exist_ok=True)

from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(
    filename=f'logs/model_comparison_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
)
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_models(device: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Load both Qwen models and their tokenizers in both native and HookedTransformer formats.

    Returns:
        models: Dictionary containing model objects
        model_info: Dictionary containing model statistics and metadata
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Model paths
    qwen_original = "Qwen/Qwen2.5-Math-1.5B"
    qwen_tuned = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Store models and related info
    models = {}
    model_info = defaultdict(dict)

    for name, path in [("original", qwen_original), ("tuned", qwen_tuned)]:
        # logging.info(f"\nLoading {name} model from {path}")

        # Load native models and tokenizers
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()
        model.to(device)
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Create HookedTransformer config
        cfg = HookedTransformerConfig(
            n_ctx=model.config.max_position_embeddings,
            d_model=model.config.hidden_size,
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            d_head=model.config.hidden_size // model.config.num_attention_heads,
            d_vocab=model.config.vocab_size,
            tie_word_embeddings=model.config.tie_word_embeddings,
            act_fn=model.config.hidden_act,
            device=device,
        )

        # Load HookedTransformer version
        # logging.info(f"Loading HookedTransformer for {name}")
        hooked_model = HookedTransformer.from_pretrained(
            path,
            config=cfg,
            fold_ln=False,  # Keep layer norm separate for analysis
            center_writing_weights=False,
            # center_unembed=False,
            center_unembed=True,
        )
        
        # Disable gradients for hooked model
        for param in hooked_model.parameters():
            param.requires_grad = False

        # Store models
        models[name] = {
            "model": model,
            "tokenizer": tokenizer,
            "hooked": hooked_model
        }

        # Collect model info
        model_info[name] = {
            "n_layers": hooked_model.cfg.n_layers,
            "d_model": hooked_model.cfg.d_model,
            "n_heads": hooked_model.cfg.n_heads,
            "d_head": hooked_model.cfg.d_head,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }

    return models, model_info


def compare_model_weights(original_model, tuned_model):
    """
    Compare weights between original and tuned models.
    Returns statistics about weight differences.
    """
    stats = defaultdict(dict)

    # Get named parameters from both models
    orig_params = dict(original_model.named_parameters())
    tuned_params = dict(tuned_model.named_parameters())

    for name in orig_params.keys():
        if name in tuned_params:
            # Convert to numpy for calculations
            orig_weights = orig_params[name].detach().cpu().numpy()
            tuned_weights = tuned_params[name].detach().cpu().numpy()

            # Calculate differences
            diff = tuned_weights - orig_weights

            # Compute statistics
            stats[name] = {
                "numel": orig_weights.size,
                "mean_diff": float(np.mean(np.abs(diff))),
                "max_diff": float(np.max(np.abs(diff))),
                "std_diff": float(np.std(diff)),
                "norm_diff": float(np.linalg.norm(diff)),
                "relative_diff": float(np.linalg.norm(diff) / np.linalg.norm(orig_weights))
            }

            # # Calculate KL divergence between weight distributions
            # # Add small epsilon to avoid division by zero
            # eps = 1e-10
            # orig_flat = orig_weights.flatten() + eps
            # tuned_flat = tuned_weights.flatten() + eps
            
            # # Normalize to create probability distributions
            # orig_dist = orig_flat / np.sum(orig_flat)
            # tuned_dist = tuned_flat / np.sum(tuned_flat)
            
            # # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
            # kl_div = np.sum(orig_dist * np.log(orig_dist / tuned_dist))
            # stats[name]["kl_divergence"] = float(kl_div)

    return stats

def analyze_weight_differences(stats, model_info):
    """
    Analyze and print insights about weight differences.
    """
    # Sort layers by relative difference
    sorted_layers = sorted(
        stats.items(),
        key=lambda x: x[1]["relative_diff"],
        reverse=True
    )

    # # Sort layers by KL divergence
    # sorted_by_kl = sorted(
    #     stats.items(),
    #     key=lambda x: x[1]["kl_divergence"],
    #     reverse=True
    # )

    logging.info("\nModel Architecture Info:")
    logging.info("-" * 40)
    for model_name, info in model_info.items():
        logging.info(f"\n{model_name.upper()} Model:")
        for k, v in info.items():
            logging.info(f"{k}: {v}")

    logging.info("\nTop 10 most changed layers (by relative difference):")
    logging.info("Layer name | Relative diff | Mean abs diff | Max abs diff | Numel")
    logging.info("-" * 80)
    for name, stat in sorted_layers[:10]:
        logging.info(f"{name.ljust(40)} | {stat['relative_diff']:.6f} | {stat['mean_diff']:.6f} | {stat['max_diff']:.6f} | {stat['numel']}")

    logging.info("\nTop most changed layers (by relative difference):")
    logging.info("Layer name | Relative diff | Mean abs diff | Max abs diff | Numel")
    logging.info("-" * 80)
    for name, stat in sorted_layers:
        logging.info(f"{name.ljust(40)} | {stat['relative_diff']:.6f} | {stat['mean_diff']:.6f} | {stat['max_diff']:.6f} | {stat['numel']}")

    logging.info("\nTop 10 least changed layers (by relative difference):")
    logging.info("Layer name | Relative diff | Mean abs diff | Max abs diff | Numel")
    logging.info("-" * 80)
    for name, stat in sorted_layers[-10:]:
        logging.info(f"{name.ljust(40)} | {stat['relative_diff']:.6f} | {stat['mean_diff']:.6f} | {stat['max_diff']:.6f} | {stat['numel']}")

    # logging.info("\nTop 10 layers by KL divergence:")
    # logging.info("Layer name | KL divergence | Relative diff | Mean abs diff | Max abs diff")
    # logging.info("-" * 80)
    # for name, stat in sorted_by_kl[:10]:
    #     logging.info(f"{name[:30]:<30} | {stat['kl_divergence']:.6f} | {stat['relative_diff']:.6f} | {stat['mean_diff']:.6f} | {stat['max_diff']:.6f}")

    # logging.info("\nLeast changed 10 layers by KL divergence:")
    # logging.info("Layer name | KL divergence | Relative diff | Mean abs diff | Max abs diff")
    # logging.info("-" * 80)
    # for name, stat in sorted_by_kl[-10:]:
    #     logging.info(f"{name[:30]:<30} | {stat['kl_divergence']:.6f} | {stat['relative_diff']:.6f} | {stat['mean_diff']:.6f} | {stat['max_diff']:.6f}")

# def get_activation_stats(model: HookedTransformer,
#                         input_text: str,
#                         layer_name: str) -> Dict:
#     """
#     Get activation statistics for a specific layer using HookedTransformer.
#     """
#     def activation_hook(value, hook):
#         hook.ctx["activations"] = value.detach()

#     ctx = {}
#     model.run_with_hooks(
#         model.to_tokens(input_text),
#         fwd_hooks=[(layer_name, activation_hook)],
#         hook_ctx=ctx
#     )

#     activations = ctx["activations"]
#     return {
#         "mean": float(t.mean(activations).item()),
#         "std": float(t.std(activations).item()),
#         "max": float(t.max(activations).item()),
#         "min": float(t.min(activations).item()),
#     }

# def compare_activations(models: Dict,
#                        input_text: str = "Let's solve this step by step:") -> Dict:
#     """
#     Compare activations between original and tuned models.
#     """
#     activation_diffs = {}

#     # Get activations for both models
#     for layer_idx in range(models["original"]["hooked"].cfg.n_layers):
#         layer_name = f"blocks.{layer_idx}.hook_resid_pre"

#         orig_stats = get_activation_stats(
#             models["original"]["hooked"],
#             input_text,
#             layer_name
#         )
#         tuned_stats = get_activation_stats(
#             models["tuned"]["hooked"],
#             input_text,
#             layer_name
#         )

#         # Calculate differences
#         activation_diffs[layer_idx] = {
#             k: abs(tuned_stats[k] - orig_stats[k])
#             for k in orig_stats.keys()
#         }

#     return activation_diffs

# def compare_kl_divergence(original_model, tuned_model, input_text: str):
#     """
#     Compare the KL divergence of the original and tuned models on a given input text.
#     List the top tokens with the highest and lowest KL divergence and visualize the results.

#     Args:
#         original_model: The original model to compare.
#         tuned_model: The tuned model to compare.
#         input_text: The input text for which to compute KL divergence.

#     Returns:
#         None
#     """
#     # Tokenize the input text
#     original_tokens = original_model.tokenizer(input_text, return_tensors='pt')
#     tuned_tokens = tuned_model.tokenizer(input_text, return_tensors='pt')

#     # Get logits from both models, detach and move to CPU
#     with t.no_grad():
#         original_logits = original_model(**original_tokens).logits.detach().cpu()
#         tuned_logits = tuned_model(**tuned_tokens).logits.detach().cpu()

#     # Compute probabilities
#     original_probs = t.softmax(original_logits, dim=-1)
#     tuned_probs = t.softmax(tuned_logits, dim=-1)

#     # Compute KL divergence
#     kl_divergence = t.nn.functional.kl_div(tuned_probs.log(), original_probs, reduction='none')

#     # Get top tokens with highest and lowest KL divergence
#     top_tokens = t.topk(kl_divergence, k=10)
#     lowest_tokens = t.topk(kl_divergence, k=10, largest=False)

#     # Visualization (simple text-based for now)
#     print("Top tokens with highest KL divergence:")
#     for token in top_tokens.indices[0]:
#         print(f"Token: {original_model.tokenizer.decode(token)}, KL Divergence: {top_tokens.values[0][token]}")

#     print("\nTop tokens with lowest KL divergence:")
#     for token in lowest_tokens.indices[0]:
#         print(f"Token: {original_model.tokenizer.decode(token)}, KL Divergence: {lowest_tokens.values[0][token]}")

#     # Visualization can be enhanced with libraries like matplotlib or seaborn
#     # Here we can create a color map based on KL divergence values
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(kl_divergence)), kl_divergence.cpu(), 
#             color=plt.cm.viridis(kl_divergence.cpu() / kl_divergence.max().cpu()))
#     plt.title('KL Divergence per Token')
#     plt.xlabel('Token Index')
#     plt.ylabel('KL Divergence')
#     plt.show()


# %%

models, model_info = load_models()
# %%

    # Compare weights
logging.info("\nComparing model weights...")
stats = compare_model_weights(
    models["original"]["model"],
    models["tuned"]["model"]
)

# Analyze differences
analyze_weight_differences(stats, model_info)

# %%

# # Compare KL divergence
# logging.info("\nComparing KL divergence...")
# compare_kl_divergence(models["original"]["model"], models["tuned"]["model"], "Let's solve this step by step:")

# # %%

# if __name__ == "__main__":
#     # Load models
#     models, model_info = load_models()

#     # Compare weights
#     logging.info("\nComparing model weights...")
#     stats = compare_model_weights(
#         models["original"]["model"],
#         models["tuned"]["model"]
#     )

#     # Analyze differences
#     analyze_weight_differences(stats, model_info)

#     # Compare activations on a sample input
#     logging.info("\nComparing activations...")
#     activation_diffs = compare_activations(models)

#     # Print activation differences
#     logging.info("\nActivation differences by layer:")
#     logging.info("Layer | Mean diff | Std diff | Max diff | Min diff")
#     logging.info("-" * 60)
#     for layer_idx, diffs in activation_diffs.items():
#         logging.info(f"{layer_idx:>5} | {diffs['mean']:.6f} | {diffs['std']:.6f} | {diffs['max']:.6f} | {diffs['min']:.6f}")

#     # Compare KL divergence
#     logging.info("\nComparing KL divergence...")
#     compare_kl_divergence(models["original"]["model"], models["tuned"]["model"], "Let's solve this step by step:")


# %%

import torch as t
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# %%
def calculate_kl_for_position(orig_probs: t.Tensor, tuned_probs: t.Tensor, pos: int) -> float:
    """Helper function to calculate KL divergence for a position"""
    # Ensure no zero probabilities (add small epsilon)
    eps = 1e-10
    p = orig_probs[pos] + eps
    q = tuned_probs[pos] + eps
    
    # Calculate KL divergence
    return t.sum(p * t.log(p / q)).item()

def compare_kl_divergence(
    models: dict,
    text: str,
    top_n: int = 5,
    visualize: bool = True
) -> Tuple[Tuple[dict, List[Tuple[str, float]], List[Tuple[str, float]]], Tuple[dict, List[Tuple[str, float]], List[Tuple[str, float]]]]:
    """
    Compare KL divergence between original and tuned models on a text sequence.
    Calculates KL divergence both for current token predictions and next-token predictions.
    
    Args:
        models: Dictionary containing original and tuned model objects
        text: Input text to analyze
        top_n: Number of top/bottom tokens to return
        visualize: Whether to display color visualization
    
    Returns:
        kl_divs_current: Dictionary with per-token KL divergences for current token predictions
        kl_divs_next: Dictionary with per-token KL divergences for next token predictions
        top_tokens: List of (token, kl_div) tuples with highest divergence
        bottom_tokens: List of (token, kl_div) tuples with lowest divergence
    """
    # Get tokenizers and hooked models
    orig_tokenizer = models["original"]["tokenizer"]
    orig_hooked = models["original"]["model"]
    tuned_hooked = models["tuned"]["model"]

    # Tokenize input
    tokens = orig_tokenizer.encode(text)
    token_strs = [orig_tokenizer.decode([tok]) for tok in tokens]
    
    # Get logits for both models
    input_tensor = t.tensor(tokens).unsqueeze(0).to(device)
    
    orig_logits = orig_hooked(input_tensor)[0].detach().cpu()
    tuned_logits = tuned_hooked(input_tensor)[0].detach().cpu()
    
    # Convert to probabilities
    orig_probs = F.softmax(orig_logits, dim=-1).squeeze(0)
    tuned_probs = F.softmax(tuned_logits, dim=-1).squeeze(0)
    
    # Calculate KL divergence for each token position
    # Track both current-token and next-token KL divergence separately
    kl_divs_current = collections.defaultdict(float)
    kl_divs_next = collections.defaultdict(float)
    token_counts = collections.defaultdict(int)
    
    # Current token KL divergence
    for pos in range(len(tokens)):
        kl = calculate_kl_for_position(orig_probs, tuned_probs, pos)
        kl_divs_current[token_strs[pos]] += kl
        token_counts[token_strs[pos]] += 1
        
    # Next token KL divergence (looking at predictions from previous token)
    for pos in range(len(tokens)-1):
        kl = calculate_kl_for_position(orig_probs, tuned_probs, pos+1)
        kl_divs_next[token_strs[pos]] += kl
    
    # Average the KL divergences by token counts
    for token in token_counts.keys():
        kl_divs_current[token] /= token_counts[token]
        if token in kl_divs_next:
            kl_divs_next[token] /= token_counts[token]

    # Get top and bottom tokens for both current and next KL
    sorted_current = sorted(kl_divs_current.items(), key=lambda x: x[1], reverse=True)
    sorted_next = sorted(kl_divs_next.items(), key=lambda x: x[1], reverse=True)
    
    # Logging results
    logging.info(f"\nKL Divergence Analysis for text: '{text}'")
    
    logging.info(f"\nTop {top_n} tokens with highest KL divergence (current token):")
    for token, kl in sorted_current[:top_n]:
        logging.info(f"Token: '{token}' | Current KL: {kl:.4f}")
    
    logging.info(f"\nTop {top_n} tokens with highest KL divergence (next token prediction):")
    for token, kl in sorted_next[:top_n]:
        logging.info(f"Token: '{token}' | Next KL: {kl:.4f}")
    
    logging.info(f"\nLowest {top_n} tokens with lowest KL divergence (current token):")
    for token, kl in sorted_current[-top_n:]:
        logging.info(f"Token: '{token}' | Current KL: {kl:.4f}")
    
    logging.info(f"\nLowest {top_n} tokens with lowest KL divergence (next token prediction):")
    for token, kl in sorted_next[-top_n:]:
        logging.info(f"Token: '{token}' | Next KL: {kl:.4f}")
    
    # Visualization
    if visualize:
        # Visualize current token KL
        logging.info("\nCurrent Token KL Divergence:")
        display_kl_visualization([t for t,_ in sorted_current], [kl for _,kl in sorted_current], "Current Token")
        display_kl_visualization([t for t,_ in sorted_current][:50], [kl for _,kl in sorted_current][:50], "Current Token (Top 50)")
        display_kl_visualization(token_strs, [kl_divs_current[t] for t in token_strs], "Current Token (Sequence)")
        
        # Visualize next token KL
        logging.info("\nNext Token Prediction KL Divergence:")
        display_kl_visualization([t for t,_ in sorted_next], [kl for _,kl in sorted_next], "Next Token")
        display_kl_visualization([t for t,_ in sorted_next][:50], [kl for _,kl in sorted_next][:50], "Next Token (Top 50)")
        display_kl_visualization(token_strs[:-1], [kl_divs_next[t] for t in token_strs[:-1]], "Next Token (Sequence)")

    return (kl_divs_current, sorted_current[:top_n], sorted_current[-top_n:]), (kl_divs_next, sorted_next[:top_n], sorted_next[-top_n:])

def display_kl_visualization(tokens: List[str], kl_values: List[float], title_prefix: str = ""):
    """
    Display tokens with color-coded KL divergence values.
    """
    # Normalize KL values for coloring (0 to 1 scale)
    kl_min, kl_max = min(kl_values), max(kl_values)
    if kl_max == kl_min:  # Avoid division by zero
        normalized = [0.5] * len(kl_values)
    else:
        normalized = [(x - kl_min) / (kl_max - kl_min) for x in kl_values]
    
    # Create HTML visualization
    html = "<div style='font-family: monospace; white-space: pre;'>"
    html += f"<h4>{title_prefix} KL Divergence</h4>"
    for token, kl_norm, kl_raw in zip(tokens, normalized, kl_values):
        # Convert normalized value to RGB (red for high, green for low)
        r = int(255 * kl_norm)
        g = int(255 * (1 - kl_norm))
        b = 0
        color = f'#{r:02x}{g:02x}{b:02x}'
        html += f"<span style='background-color: {color}; padding: 2px; margin: 1px;' title='KL: {kl_raw:.4f}'>{token}</span>"
    html += "</div>"
    
    display(HTML(html))
    
    # Also create a simple plot
    plt.figure(figsize=(12, 4))
    plt.plot(kl_values, 'b-', label='KL Divergence')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel('KL Divergence')
    plt.title(f'{title_prefix} KL Divergence per Token')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%

# Example usage in your main block
if MAIN:
    # Load models
    models, model_info = load_models()
    
    
# %%
# Example text
# sample_text = "The quick brown fox jumps over the lazy dog"

with open("experiment_outputs/2025-02-25_05-03-38/groq_deepseek_32b/problem_0/dropout/prop_1.0/prompt.txt", "r") as f:
    sample_text = f.read()

# Compare KL divergence
(kl_divs_current, top_tokens_current, bottom_tokens_current), (kl_divs_next, top_tokens_next, bottom_tokens_next) = compare_kl_divergence(
    models,
    sample_text,
    top_n=5,
    visualize=True
)

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    suffix = t.randint(high=len(model.tokenizer) - 1, size=(batch_size, seq_len - 1)).long()
    return t.cat([prefix, suffix], dim=-1)

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens, logits, cache). This
    function should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_tokens = t.cat([rep_tokens, rep_tokens], dim=-1)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=True)
    return rep_tokens, rep_logits, rep_cache

def induction_attn_detector(model: HookedTransformer, cache: ActivationCache, threshold: float = 0.01) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer].squeeze(0)[head]
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > threshold:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

# %%
import matplotlib.pyplot as plt
import numpy as np

thresholds = np.linspace(0.01, 0.1, 20)
results = {model_type: [] for model_type in ["original", "tuned"]}

for model_type in ["original", "tuned"]:
    model, seq_len, batch_size = models[model_type]["hooked"], 50, 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
    rep_cache.remove_batch_dim()
    
    total_heads = model.cfg.n_layers * model.cfg.n_heads
    for threshold in thresholds:
        num_induction_heads = len(induction_attn_detector(model, rep_cache, threshold))
        results[model_type].append(num_induction_heads / total_heads)

plt.figure(figsize=(10, 6))
for model_type, scores in results.items():
    plt.plot(thresholds, scores, label=model_type, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Fraction of Heads Classified as Induction Heads')
plt.title('Induction Head Detection at Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()


# %%


# %%
