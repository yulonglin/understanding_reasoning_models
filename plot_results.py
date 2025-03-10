import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Font size constant for plots
PLOT_FONT_SIZE = 21  # 10x larger than default 12

def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs("figures", exist_ok=True)


def find_latest_experiment_dirs():
    """Find the latest directory for each experiment type."""
    base_dir = "experiment_outputs"
    date_dirs = [d for d in os.listdir(base_dir) if d[0].isdigit()]
    date_dirs.sort(reverse=True)  # Sort by date, newest first

    latest_results = {}
    for date_dir in date_dirs:
        full_date_dir = os.path.join(base_dir, date_dir)
        for model_dir in os.listdir(full_date_dir):
            model_path = os.path.join(full_date_dir, model_dir)
            if not os.path.isdir(model_path):
                continue

            for result_file in os.listdir(model_path):
                if result_file.endswith("_results.json"):
                    experiment_type = result_file.replace("_results.json", "")
                    if experiment_type not in latest_results:
                        latest_results[experiment_type] = (
                            date_dir,
                            model_dir,
                            os.path.join(model_path, result_file),
                        )

    return latest_results


def plot_determinism_results(results_file):
    """Plot determinism experiment results."""
    with open(results_file) as f:
        results = json.load(f)

    most_common_freqs = [r["most_common_freq"] for r in results]
    unique_counts = [r["unique_answers"] for r in results]

    # Calculate and plot average generation length for each problem
    date_dir, model_dir = os.path.dirname(results_file).split(os.sep)[-2:]
    print(date_dir, model_dir)
    base_path = os.path.join(os.getcwd(), "experiment_outputs", date_dir, model_dir)
    
    avg_lengths = []
    std_lengths = []
    
    for problem_idx in range(len(results)):
        problem_dir = os.path.join(base_path, f"problem_{problem_idx}", "determinism")
        print(problem_dir)
        print(os.getcwd())
        print(os.path.exists(problem_dir))
        if os.path.exists(problem_dir):
            gen_files = [f for f in os.listdir(problem_dir) if f.startswith("gen_") and f.endswith(".out")]
            lengths = []
            
            for gen_file in gen_files:
                file_path = os.path.join(problem_dir, gen_file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    lengths.append(len(content))
            
            print(lengths)
            if lengths:
                avg_lengths.append(np.mean(lengths))
                std_lengths.append(np.std(lengths))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)
    
    # Create a single figure with 3 subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot frequency of most common answer
    ax1.bar(range(len(most_common_freqs)), most_common_freqs)
    ax1.set_title("Frequency of Most Common Answer", fontsize=PLOT_FONT_SIZE)
    ax1.set_xlabel("Problem Index", fontsize=PLOT_FONT_SIZE)
    ax1.set_ylabel("Count (out of 10)", fontsize=PLOT_FONT_SIZE)

    # Plot number of unique answers
    ax2.bar(range(len(unique_counts)), unique_counts)
    ax2.set_title("Number of Unique Answers", fontsize=PLOT_FONT_SIZE)
    ax2.set_xlabel("Problem Index", fontsize=PLOT_FONT_SIZE)
    ax2.set_ylabel("Count", fontsize=PLOT_FONT_SIZE)
    
    # Plot average generation length
    ax3.bar(range(len(avg_lengths)), avg_lengths, yerr=std_lengths, capsize=5)
    ax3.set_title("Average Generation Length", fontsize=PLOT_FONT_SIZE)
    ax3.set_xlabel("Problem Index", fontsize=PLOT_FONT_SIZE)
    ax3.set_ylabel("Average Length (characters)", fontsize=PLOT_FONT_SIZE)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("figures/determinism_results.png")
    plt.close()


def plot_truncation_results(results_file):
    """Plot truncation experiment results."""
    with open(results_file) as f:
        results = json.load(f)

    proportions = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Plot aggregate results
    avg_correct = []
    std_correct = []

    for i, prop in enumerate(proportions):
        correct_counts = [r["results"][i]["correct_count"] for r in results]
        avg_correct.append(np.mean(correct_counts))
        std_correct.append(np.std(correct_counts))

    plt.figure(figsize=(10, 6))
    plt.plot(proportions, avg_correct, 'o-')
    plt.fill_between(proportions, 
                     [avg_correct[i] - std_correct[i] for i in range(len(avg_correct))],
                     [avg_correct[i] + std_correct[i] for i in range(len(avg_correct))],
                     alpha=0.3)
    plt.title("Effect of CoT Truncation on Performance (Aggregate)", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Proportion of CoT Steps Kept", fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Average Correct Answers (out of 10)", fontsize=PLOT_FONT_SIZE)
    plt.grid(True)
    plt.savefig("figures/truncation_results_aggregate.png")
    plt.close()

    # Plot individual problem results
    plt.figure(figsize=(12, 8))
    for problem_idx, problem_result in enumerate(results):
        correct_counts = [r["correct_count"] for r in problem_result["results"]]
        plt.plot(proportions, correct_counts, "o-", label=f"Problem {problem_idx}")

    plt.title("Effect of CoT Truncation on Performance (Individual Problems)", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Proportion of CoT Steps Kept", fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Correct Answers (out of 10)", fontsize=PLOT_FONT_SIZE)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig("figures/truncation_results_individual.png", bbox_inches="tight")
    plt.close()


def plot_dropout_results(results_file):
    """Plot dropout experiment results."""
    with open(results_file) as f:
        results = json.load(f)

    proportions = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Plot aggregate results
    avg_correct = []
    std_correct = []

    for i, prop in enumerate(proportions):
        correct_counts = [r["results"][i]["correct_count"] for r in results]
        avg_correct.append(np.mean(correct_counts))
        std_correct.append(np.std(correct_counts))

    plt.figure(figsize=(10, 6))
    plt.plot(proportions, avg_correct, 'o-')
    plt.fill_between(proportions, 
                     [avg_correct[i] - std_correct[i] for i in range(len(avg_correct))],
                     [avg_correct[i] + std_correct[i] for i in range(len(avg_correct))],
                     alpha=0.3)
    plt.title("Effect of CoT Dropout on Performance (Aggregate)", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Proportion of CoT Steps Kept", fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Average Correct Answers (out of 10)", fontsize=PLOT_FONT_SIZE)
    plt.grid(True)
    plt.savefig("figures/dropout_results_aggregate.png")
    plt.close()

    # Plot individual problem results
    plt.figure(figsize=(12, 8))
    for problem_idx, problem_result in enumerate(results):
        correct_counts = [r["correct_count"] for r in problem_result["results"]]
        plt.plot(proportions, correct_counts, "o-", label=f"Problem {problem_idx}")

    plt.title("Effect of CoT Dropout on Performance (Individual Problems)", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Proportion of CoT Steps Kept", fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Correct Answers (out of 10)", fontsize=PLOT_FONT_SIZE)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig("figures/dropout_results_individual.png", bbox_inches="tight")
    plt.close()


def plot_shuffling_results(results_file):
    """Plot shuffling experiment results."""
    with open(results_file) as f:
        results = json.load(f)

    correct_counts = [r["correct_count"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(correct_counts)), correct_counts)
    plt.title("Effect of CoT Shuffling on Performance", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Problem Index", fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Correct Answers (out of 10)", fontsize=PLOT_FONT_SIZE)
    plt.grid(True)
    plt.savefig("figures/shuffling_results.png")
    plt.close()


def main():
    ensure_figures_dir()
    latest_results = find_latest_experiment_dirs()

    plot_functions = {
        "determinism": plot_determinism_results,
        "truncation": plot_truncation_results,
        "dropout": plot_dropout_results,
        "shuffling": plot_shuffling_results,
    }

    for experiment_type, (date_dir, model_dir, results_file) in latest_results.items():
        print(f"Plotting {experiment_type} results from {date_dir}/{model_dir}")
        plot_functions[experiment_type](results_file)


if __name__ == "__main__":
    main()
