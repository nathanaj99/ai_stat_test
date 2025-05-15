import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess(df_results, df_results_original):
    dropped = set(df_results_original['question_id']) - set(df_results['question_id'])
    df_results_original = df_results_original[~df_results_original['question_id'].isin(dropped)]
    perturbed_results = np.vstack(df_results.groupby('question_id')['correct'].apply(lambda x: x.values))
    original_results = df_results_original['correct'].to_numpy()

    return perturbed_results, original_results

## \bar{s} diagram
def bar_s_bootstrap(original_results, perturbed_results, alpha=0.05):
    diffs = []
    for i in range(len(original_results)):
        diff = original_results[i] - np.mean(perturbed_results[i])
        diffs.append(diff)
    diffs = np.array(diffs)

    B = 1000
    rng = np.random.default_rng(42)
    s_boot = np.empty(B)
    for b in range(B):
        # Resample question indices with replacement
        indices = rng.integers(0, len(diffs), size=len(diffs))
        s_boot[b] = np.mean(diffs[indices])

    # Calculate 95% confidence intervals from the bootstrap distribution
    ci_lower, ci_upper = np.percentile(s_boot, [alpha*100/2, 100-alpha*100/2])
    return np.mean(s_boot), ci_lower, ci_upper, diffs

means = {}
ci_lowers = {}
ci_uppers = {}
list_of_diffs = {}
perturbed_results = {}
original_results = {}

model_list = ['gpt-4.1', 'gpt-4.1-mini']


for model_id in model_list:
    means[model_id] = {}
    ci_lowers[model_id] = {}
    ci_uppers[model_id] = {}
    list_of_diffs[model_id] = {}
    perturbed_results[model_id] = {}
    original_results[model_id] = {}

    dataset = 'movie_recommendation'
    df_results = pd.read_csv(f'../../files/bbh/results/movie_recommendation/{model_id}/perturbed_with_correct.csv')
    df_results_original = pd.read_csv(f'../../files/bbh/results/movie_recommendation/{model_id}/original_with_correct.csv')
    perturbed_results_buffer, original_results_buffer = preprocess(df_results, df_results_original)
    mean, lower, upper, diffs = bar_s_bootstrap(original_results_buffer, perturbed_results_buffer, alpha=0.05)

    means[model_id][dataset] = mean
    ci_lowers[model_id][dataset] = lower
    ci_uppers[model_id][dataset] = upper
    list_of_diffs[model_id][dataset] = diffs
    perturbed_results[model_id][dataset] = perturbed_results_buffer
    original_results[model_id][dataset] = original_results_buffer

    dataset = 'GPQA'
    df_results = pd.read_csv(f'../../files/bbh/results/GPQA/{model_id}/perturbed_with_correct.csv')
    df_results_original = pd.read_csv(f'../../files/bbh/results/GPQA/{model_id}/original_with_correct.csv')
    perturbed_results_buffer, original_results_buffer = preprocess(df_results, df_results_original)
    mean, lower, upper, diffs = bar_s_bootstrap(original_results_buffer, perturbed_results_buffer, alpha=0.05)

    means[model_id][dataset] = mean
    ci_lowers[model_id][dataset] = lower
    ci_uppers[model_id][dataset] = upper
    list_of_diffs[model_id][dataset] = diffs
    perturbed_results[model_id][dataset] = perturbed_results_buffer
    original_results[model_id][dataset] = original_results_buffer



plt.rcParams.update({'font.size': 24})
datasets = ['MR', 'GPQA']
datasets_raw = ['movie_recommendation', 'GPQA']
positions = np.arange(1, len(datasets) + 1)  # [1, 2]

fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), sharey=False)

# --- First two: errorbar plots ---
for i, model_id in enumerate(model_list):
    means_list = []
    ci_lowers_list = []
    ci_uppers_list = []
    for dataset in datasets_raw:
        means_list.append(means[model_id][dataset])
        ci_lowers_list.append(ci_lowers[model_id][dataset])
        ci_uppers_list.append(ci_uppers[model_id][dataset])
    axes[i].errorbar(
        [0, 1], means_list, 
        yerr=[np.array(means_list) - np.array(ci_lowers_list), 
              np.array(ci_uppers_list) - np.array(means_list)],
        fmt='o', capsize=6, capthick=2, markersize=8,
        color='blue', ecolor='black', label='95% CI'
    )
    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_ylabel(r'Avg. bias ($\bar{s}$)')
    if i == 1:
        axes[i].set_yticklabels([])
    axes[i].set_xticks([0, 1])
    axes[i].set_xlim(-0.5, 1.5)
    axes[i].set_xticklabels(datasets)
    axes[i].set_title(model_id)
    axes[i].set_ylim(-0.15, 0.09)

# --- Next two: boxplots ---
for i, model_id in enumerate(model_list):
    box_data = []
    for j, (dataset, dataset_raw) in enumerate(zip(datasets, datasets_raw)):
        pert = perturbed_results[model_id][dataset_raw]
        row_means = np.mean(pert, axis=1, keepdims=True)
        abs_diffs = np.abs(pert - row_means)
        row_avg_abs_diffs = np.mean(abs_diffs, axis=1)
        box_data.append(row_avg_abs_diffs)
    ax_box = axes[i+2]
    bp = ax_box.boxplot(box_data, patch_artist=True, positions=positions)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(datasets)
    ax_box.tick_params(axis='y')
    ax_box.grid(True, alpha=0.3)
    if i == 0:
        ax_box.set_ylabel(r'$M_i$ dist.')
    if i == 1:
        ax_box.set_yticklabels([])
    ax_box.set_title(model_id)
    ax_box.set_ylim(-0.05, 0.55)

# --- Add (a) and (b) labels ---
axes[0].annotate('(a)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=24, fontweight='bold', xytext=(-60, 16), textcoords='offset points', ha='left', va='top')
axes[2].annotate('(b)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=24, fontweight='bold', xytext=(-60, 16), textcoords='offset points', ha='left', va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.05)
# Get current positions
pos1 = axes[1].get_position()
pos2 = axes[2].get_position()
pos3 = axes[3].get_position()

# Amount to shift (adjust as needed)
shift = 0.06

# Move axes[2] and axes[3] to the right
axes[2].set_position([pos2.x0 + shift, pos2.y0, pos2.width, pos2.height])
axes[3].set_position([pos3.x0 + shift, pos3.y0, pos3.width, pos3.height])
plt.savefig('../../figs/gpt4.1_analysis.png', bbox_inches='tight')
plt.show()