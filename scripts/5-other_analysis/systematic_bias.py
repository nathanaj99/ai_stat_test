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

model_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']
dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

means = {}
ci_lowers = {}
ci_uppers = {}
list_of_diffs = {}
perturbed_results = {}
original_results = {}

for i, model_id in enumerate(model_list):
    means[model_id] = {}
    ci_lowers[model_id] = {}
    ci_uppers[model_id] = {}
    list_of_diffs[model_id] = {}
    perturbed_results[model_id] = {}
    original_results[model_id] = {}
    for cat in dataset_dic:
        if cat == 'lmentry':
            fn = '500_20'
        else:
            fn = '20'
        for dataset in dataset_dic[cat]:
            df_results = pd.read_csv(f'../../files/{cat}/results/{dataset}/{model_id}/{fn}_with_correct.csv')
            df_results_original = pd.read_csv(f'../../files/{cat}/results/{dataset}/{model_id}/original_with_correct.csv')
            perturbed_results_buffer, original_results_buffer = preprocess(df_results, df_results_original)
            mean, lower, upper, diffs = bar_s_bootstrap(original_results_buffer, perturbed_results_buffer)
            means[model_id][dataset] = mean
            ci_lowers[model_id][dataset] = lower
            ci_uppers[model_id][dataset] = upper
            list_of_diffs[model_id][dataset] = diffs
            perturbed_results[model_id][dataset] = perturbed_results_buffer
            original_results[model_id][dataset] = original_results_buffer

## Main figure
plt.rcParams.update({'font.size': 22})

fig, axes = plt.subplots(2, 3, figsize=(20, 6), sharex='col', sharey='row')

model_list = ['Llama-3.2-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-4b-it']
dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']
datasets_raw = ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word', 'causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks']

positions = np.arange(1, len(datasets) + 1)  # [1, 2, 3, 4]


# --- First row: errorbar plots ---
for i, model_id in enumerate(model_list):
    means_list = []
    ci_lowers_list = []
    ci_uppers_list = []
    for cat in dataset_dic:
        for dataset in dataset_dic[cat]:
            means_list.append(means[model_id][dataset])
            ci_lowers_list.append(ci_lowers[model_id][dataset])
            ci_uppers_list.append(ci_uppers[model_id][dataset])
    axes[0, i].errorbar(
        positions, means_list, 
        yerr=[np.array(means_list) - np.array(ci_lowers_list), 
              np.array(ci_uppers_list) - np.array(means_list)],
        fmt='o', capsize=6, capthick=2, markersize=8,
        color='blue', ecolor='black', label='95% CI'
    )
    axes[0, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, i].grid(True, alpha=0.3)
    if i == 0:
        axes[0, i].set_ylabel(r'Avg bias ($\bar{s}$)')
    axes[0, i].set_title(model_id)
    axes[0, i].set_xticks(positions)
    axes[0, i].set_xticklabels(datasets)  # Only show on bottom row, but set for alignment

# --- Second row: boxplots ---
for i, model_id in enumerate(model_list):
    box_data = []
    for j, (dataset, dataset_raw) in enumerate(zip(datasets, datasets_raw)):
        pert = perturbed_results[model_id][dataset_raw]
        row_means = np.mean(pert, axis=1, keepdims=True)
        abs_diffs = np.abs(pert - row_means)
        row_avg_abs_diffs = np.mean(abs_diffs, axis=1)
        box_data.append(row_avg_abs_diffs)
    bp = axes[1, i].boxplot(box_data, patch_artist=True, positions=positions)
    axes[1, i].set_xticks(positions)
    axes[1, i].set_xticklabels(datasets)
    axes[1, i].tick_params(axis='y')
    axes[1, i].grid(True, alpha=0.3)
    if i == 0:
        axes[1, i].set_ylabel(r'$M_i$ dist.')

# --- Add (a) and (b) labels ---
axes[0, 0].annotate('(a)', xy=(-0.05, 1), xycoords='axes fraction', fontsize=22, fontweight='bold', xytext=(-75, 13), textcoords='offset points', ha='left', va='top')
axes[1, 0].annotate('(b)', xy=(-0.05, 0.9), xycoords='axes fraction', fontsize=22, fontweight='bold', xytext=(-75, 13), textcoords='offset points', ha='left', va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.03, hspace=0.05)
plt.savefig('../../figs/main_bias_results.png', bbox_inches='tight')
plt.show()

#### ------------------------------------------------------------
#### Appendix Figures
#### ------------------------------------------------------------


fig, ax = plt.subplots(2, 4, figsize=(25, 8), sharex=True, sharey=True)
# plt.subplots_adjust(hspace=0.3)

plt.rcParams.update({'font.size': 24})

model_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']
dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

for i, model_id in enumerate(model_list):
    means_list = []
    ci_lowers_list = []
    ci_uppers_list = []
    list_of_diffs_list = []
    perturbed_results_list = []
    original_results_list = []
    for cat in dataset_dic:
        for dataset in dataset_dic[cat]:
            means_list.append(means[model_id][dataset])
            ci_lowers_list.append(ci_lowers[model_id][dataset])
            ci_uppers_list.append(ci_uppers[model_id][dataset])
            list_of_diffs_list.append(list_of_diffs[model_id][dataset])
            perturbed_results_list.append(perturbed_results[model_id][dataset])
            original_results_list.append(original_results[model_id][dataset])

    # Determine subplot position
    row = i // 4
    col = i % 4
    if row == 1 and col == 3:
        continue  # skip the unused 8th subplot
    ax_ = ax[row, col]

    # Plot points and error bars
    ax_.errorbar(datasets, means_list, 
                yerr=[np.array(means_list) - np.array(ci_lowers_list), 
                    np.array(ci_uppers_list) - np.array(means_list)],
                fmt='o', 
                capsize=5,
                capthick=2,
                markersize=7,
                color='blue',
                ecolor='black',
                label='95% CI')

    # Add horizontal line at y=0
    ax_.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_.tick_params(axis='x', labelsize=24)
    ax_.tick_params(axis='y', labelsize=24)

    # Customize the plot
    ax_.grid(True, alpha=0.3)
    if i in [0, 4]:
        ax_.set_ylabel(r'Avg. bias ($\bar{s}$)', fontsize=24)
    ax_.set_title(model_id)

# Hide the unused 8th subplot
ax[1, 3].axis('off')
ax[0, 3].tick_params(axis='x', labelsize=24, labelbottom=True)
plt.tight_layout()
plt.subplots_adjust(wspace=0.03)
plt.savefig('../../figs/hat_s_i_diagram.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(2, 4, figsize=(25, 8), sharex=True, sharey=True)
# plt.subplots_adjust(hspace=0.3)

plt.rcParams.update({'font.size': 24})

model_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']
dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets_raw = ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word',
                'causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks']
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

for i, model_id in enumerate(model_list):
    box_data = []
    for j, (dataset, dataset_raw) in enumerate(zip(datasets, datasets_raw)):
        pert = perturbed_results[model_id][dataset_raw]
        row_means = np.mean(pert, axis=1, keepdims=True)
        abs_diffs = np.abs(pert - row_means)
        row_avg_abs_diffs = np.mean(abs_diffs, axis=1)
        box_data.append(row_avg_abs_diffs)

    # Determine subplot position
    row = i // 4
    col = i % 4
    if row == 1 and col == 3:
        continue  # skip the unused 8th subplot
    ax_ = ax[row, col]
    # print([arr.shape for arr in box_data])
    # print(len(box_data))

    bp = ax_.boxplot(box_data, patch_artist=True)
    ax_.set_xticks(np.arange(1, len(datasets) + 1))
    ax_.set_xticklabels(datasets)
    ax_.tick_params(axis='y')
    ax_.grid(True, alpha=0.3)
    # Customize the plot
    ax_.grid(True, alpha=0.3)
    if i in [0, 4]:
        ax_.set_ylabel(r'$M_i$ dist.', fontsize=24)
    ax_.set_title(model_id)

# Hide the unused 8th subplot
ax[1, 3].axis('off')
ax[0, 3].tick_params(axis='x', labelsize=24, labelbottom=True)
plt.tight_layout()
plt.subplots_adjust(wspace=0.03)
plt.savefig('../../figs/d_i_dist.png', bbox_inches='tight')
plt.show()

# change in leaderboard rankings

dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets_raw = ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word',
                'causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks']
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

x = ['Llama-1B', 'Llama-3B', 'Qwen-1B', 'Qwen-3.5B', 'Qwen-7B', 'Gemma-1B', 'Gemma-4B']
plt.rcParams['font.size'] = 25
bar_colors = [
    '#9E99FA',  # blue
    '#4941D7',  # orange
    '#FA9999',  # green
    '#F45858',  # red
    '#C21E1E',  # purple
    '#DBF681',  # brown
    '#ADD331',  # pink
]
color_map = {label: bar_colors[i % len(bar_colors)] for i, label in enumerate(x)}

fig, ax = plt.subplots(2, 8, figsize=(25, 6), sharey=True)

idx = 0
# --- First plot (original bar plot) ---
for cat in dataset_dic:
    for dataset in dataset_dic[cat]:
        bootstrap_mean_all = []
        bootstrap_ci_all = []
        for i in range(7):
            original_result = original_results[model_list[i]][dataset]
            bootstrap_samples = np.random.choice(original_result, size=(1000, len(ar)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            bootstrap_mean_all.append(np.mean(bootstrap_means))
            bootstrap_ci_all.append(np.percentile(bootstrap_means, [2.5, 97.5]))    

        theta_all = np.array(bootstrap_mean_all)
        ci_all = np.array(bootstrap_ci_all)  # shape (N, 2)

        # Calculate error bars: distance from theta to lower and upper
        yerr = np.zeros((2, len(theta_all)))
        yerr[0] = theta_all - ci_all[:, 0]  # lower error
        yerr[1] = ci_all[:, 1] - theta_all  # upper error

        # Assign colors to bars based on x-labels
        bar_color_list = [color_map[label] for label in x]
        ax[0, idx].axhline(y=0, color='black', linewidth=0.5)
        ax[0, idx].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k')
        if idx == 0:
            ax[0, idx].set_ylabel('Accuracy\n(Original)', fontsize=28)
        ax[0, idx].set_title(datasets[idx], fontsize=25)
        ax[0, idx].set_xticks([])
        ax[0, idx].grid(True, alpha=0.3)
        idx += 1

idx = 0
# --- Second plot (original errorbar plot) ---
for cat in dataset_dic:
    for dataset in dataset_dic[cat]:
        bootstrap_mean_all = []
        bootstrap_ci_all = []
        for i in range(7):
            perturbed_result = perturbed_results[model_list[i]][dataset]
            ar = np.mean(perturbed_result, axis=1)
            bootstrap_samples = np.random.choice(ar, size=(1000, len(ar)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            bootstrap_mean_all.append(np.mean(bootstrap_means))
            bootstrap_ci_all.append(np.percentile(bootstrap_means, [2.5, 97.5]))    

        theta_all = np.array(bootstrap_mean_all)
        ci_all = np.array(bootstrap_ci_all)  # shape (N, 2)

        # Calculate error bars: distance from theta to lower and upper
        yerr = np.zeros((2, len(theta_all)))
        yerr[0] = theta_all - ci_all[:, 0]  # lower error
        yerr[1] = ci_all[:, 1] - theta_all  # upper error

        # Assign colors to bars based on x-labels
        bar_color_list = [color_map[label] for label in x]
        ax[1, idx].axhline(y=0, color='black', linewidth=0.5)
        ax[1, idx].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k')
        if idx == 0:
            ax[1, idx].set_ylabel('Accuracy\n(Perturbed)', fontsize=28)
        # ax[1, idx].set_title(datasets[idx], fontsize=25)
        ax[1, idx].set_xticks([])
        ax[1, idx].grid(True, alpha=0.3)
        idx += 1

# Shared legend
plt.subplots_adjust(wspace=0.05, hspace=0.05)
handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in x]
fig.legend(handles, x, title="Model", ncol=len(x), bbox_to_anchor=(0.5, -0.1), loc='lower center', fontsize=20)
plt.savefig('../../figs/leaderboard_accuracy.png', bbox_inches='tight')
plt.show()