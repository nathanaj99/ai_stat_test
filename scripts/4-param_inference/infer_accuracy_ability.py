import numpy as np
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from scipy.special import expit
from scipy.stats import norm

## ALGORITHM 2: Adaptive Testing
def logistic(a, b, theta):
    z = a * (theta - b)
    return 1.0 / (1.0 + np.exp(-z))

def fisher_info(a, b, theta):
    p = logistic(a, b, theta)
    return a**2 * p * (1 - p)

def update_theta(theta0, a_vec, b_vec, x_vec, max_iter=5, tol=1e-4, min_info=1e-10):
    theta = theta0
    num_iter = 0
    # print(a_vec, b_vec)
    for _ in range(max_iter):
        p = logistic(a_vec, b_vec, theta)
        score = np.sum(20* a_vec * (x_vec - p))
        info  = np.sum(20* a_vec**2 * p * (1 - p))
        if info < min_info:
            print(f"Warning: info too small (info={info:.2e}) at iter {num_iter}. Stopping update.")
            return theta0, np.nan
        step  = score / info
        theta += step
        if abs(step) < tol: break
        num_iter += 1
    se = np.inf if info < min_info else 1 / np.sqrt(info)
    # print(f"num_iter: {num_iter}")
    
    return theta, se

def adaptive_test(X, a, b, max_items=10, se_target=0.3):
    n_items = len(a)
    asked, discarded, x, a_used, b_used = [], [], [], [], []
    theta_history, ci_history, se_history = [], [], []
    theta, se = 0.0, np.inf
    for _ in range(max_items):
        # select unasked item with max information at current θ
        idx = np.argmax([fisher_info(a[i], b[i], theta) if i not in asked and i not in discarded else -np.inf
                         for i in range(n_items)])
        # print(idx)
        # --- administer item here and record response ---
        response = X[idx].mean()         # returns 0/1
        asked.append(idx)
        # update records
        a_used.append(a[idx]); b_used.append(b[idx]); x.append(response)
        theta, se = update_theta(theta, np.array(a_used), np.array(b_used), np.array(x))
        if np.isnan(se):
            asked = asked[:-1]
            a_used = a_used[:-1]
            b_used = b_used[:-1]
            x = x[:-1]
            discarded.append(idx)
            continue

        # print(asked)
        if len(se_history) > 0:
            if se_history[-1] - se < se_target:
                break

        theta_history.append(theta)
        ci_history.append((theta - 1.96 * se, theta + 1.96 * se))
        se_history.append(se)
    ci = (theta - 1.96 * se, theta + 1.96 * se)
    return theta, ci, asked, theta_history, ci_history

dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}

datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

theta_results = {}
ci_results = {}
n_questions_results = {}
for cat in dataset_dic:
    for dataset in dataset_dic[cat]:
        with open(f'../../files/{cat}/data/{dataset}/irt_perturbed_params.pkl', 'rb') as f:
            irt_params = pickle.load(f)
        
        theta_all = []
        ci_all = []
        n_questions = []
        for i in range(7):
            theta, ci, asked, theta_history, ci_history = adaptive_test(
                irt_params['perturbed_results'][:, :, i],
                irt_params['a_hat'],
                irt_params['b_hat'],
                max_items=500,
                se_target=0.0001
            )
            theta_history = np.array(theta_history)
            ci_history = np.array(ci_history)  # shape (T, 2)

            lower = ci_history[:, 0]
            upper = ci_history[:, 1]
            steps = np.arange(len(theta_history))
            
            theta_all.append(theta)
            ci_all.append(ci)
            n_questions.append(len(set(asked)))

        theta_all = np.array(theta_all)
        ci_all = np.array(ci_all)  # shape (N, 2)

        theta_results[dataset] = theta_all
        ci_results[dataset] = ci_all
        n_questions_results[dataset] = n_questions

# Visualize results
datasets_raw = ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word',
                'causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks']
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

x = ['Llama-1B', 'Llama-3B', 'Qwen-1B', 'Qwen-3.5B', 'Qwen-7B', 'Gemma-1B', 'Gemma-4B']

bar_colors = [
    '#9E99FA',
    '#4941D7', 
    '#FA9999', 
    '#F45858', 
    '#C21E1E', 
    '#DBF681', 
    '#ADD331', 
]
color_map = {label: bar_colors[i % len(bar_colors)] for i, label in enumerate(x)}

plt.rcParams['font.size'] = 25
fig, ax = plt.subplots(1, 8, figsize=(25, 5))
# Make the first two subplots share y, and the last two share y
ax[1].sharey(ax[0])
ax[2].sharey(ax[0])
ax[3].sharey(ax[0])

ax[5].sharey(ax[4])
ax[6].sharey(ax[4])
ax[7].sharey(ax[4])
idx = 0
for dataset, dataset_raw in zip(datasets, datasets_raw):
    theta_all = theta_results[dataset_raw]
    ci_all = ci_results[dataset_raw]
    n_questions = n_questions_results[dataset_raw]

    # Calculate error bars: distance from theta to lower and upper
    yerr = np.zeros((2, len(theta_all)))
    yerr[0] = theta_all - ci_all[:, 0]  # lower error
    yerr[1] = ci_all[:, 1] - theta_all  # upper error

    # Assign colors to bars based on x-labels
    bar_color_list = [color_map[label] for label in x]
    ax[idx].axhline(y=0, color='black', linewidth=0.5)
    bars = ax[idx].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k', error_kw={'linewidth': 3})
    if idx == 0:
        ax[idx].set_ylabel(r'Ability $\theta$', fontsize=28)
    ax[idx].set_title(dataset)
    ax[idx].set_xticks([])

    # Annotate each bar with n_questions
    for i, bar in enumerate(bars):
        scale = 0.05
        if idx in [0, 1] and i in [2, 5]:
            scale = 0.2

        height = bar.get_height()
        # If bar is positive, place label above; if negative, below
        if height >= 0:
            va = 'bottom'
            y = height + scale * (ax[idx].get_ylim()[1] - ax[idx].get_ylim()[0])
        else:
            va = 'top'
            y = height - scale * (ax[idx].get_ylim()[1] - ax[idx].get_ylim()[0])

        if idx == 0 and i == 5:
            va = 'bottom'
        

        ax[idx].text(
            bar.get_x() + bar.get_width() / 2,
            y,
            str(n_questions[i]),
            ha='center',
            va=va,
            fontsize=20,
            fontweight='bold'
        )
    idx += 1

for idx in [1, 2, 3, 5, 6, 7]:
    ax[idx].tick_params(labelleft=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.05)
# Get current positions
pos1 = ax[1].get_position()
pos2 = ax[2].get_position()
pos4 = ax[4].get_position()
pos5 = ax[5].get_position()
pos6 = ax[6].get_position()
pos7 = ax[7].get_position()

# Amount to shift (adjust as needed)
shift = 0.025

# Move axes[2] and axes[3] to the right
ax[4].set_position([pos4.x0 + shift, pos4.y0, pos4.width, pos4.height])
ax[5].set_position([pos5.x0 + shift, pos5.y0, pos5.width, pos5.height])
ax[6].set_position([pos6.x0 + shift, pos6.y0, pos4.width, pos6.height])
ax[7].set_position([pos7.x0 + shift, pos7.y0, pos5.width, pos7.height])

# Create a shared legend for the x-labels/colors
handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in x]
fig.legend(handles, x, title="Model", ncol=len(x), bbox_to_anchor=(0.5, -0.17), loc='lower center', fontsize=22)

# plt.suptitle('Theta Estimates with Confidence Intervals')
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2)
plt.savefig('../../figs/irt_theta_estimates.png', bbox_inches='tight')


## ALGORITHM 1: Boostrap
dataset_dic = {'lmentry': ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word'],
            'bbh': ['causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks'],}
datasets_raw = ['all_words_from_category', 'first_alphabetically', 'more_letters', 'rhyming_word',
                'causal_judgment', 'movie_recommendation', 'formal_fallacies', 'snarks']
datasets = ['AWFC', 'FA', 'ML', 'RW', 'CJ', 'MR', 'FF', 'S']

x = ['Llama-1B', 'Llama-3B', 'Qwen-1B', 'Qwen-3.5B', 'Qwen-7B', 'Gemma-1B', 'Gemma-4B']
plt.rcParams['font.size'] = 25
bar_colors = [
    '#9E99FA',
    '#4941D7', 
    '#FA9999', 
    '#F45858', 
    '#C21E1E', 
    '#DBF681', 
    '#ADD331', 
]
color_map = {label: bar_colors[i % len(bar_colors)] for i, label in enumerate(x)}

fig, ax = plt.subplots(1, 8, figsize=(25, 4), sharey=True)
idx = 0

for cat in dataset_dic:
    for dataset in dataset_dic[cat]:
        with open(f'../../files/{cat}/data/{dataset}/irt_perturbed_params.pkl', 'rb') as f:
            irt_params = pickle.load(f)

        bootstrap_mean_all = []
        bootstrap_ci_all = []
        for i in range(7):
            ar = np.mean(irt_params['perturbed_results'][:, :, i], axis=1)
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
        ax[idx].axhline(y=0, color='black', linewidth=0.5)
        ax[idx].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k')
        if idx == 0:
            ax[idx].set_ylabel('Accuracy', fontsize=28)
        ax[idx].set_title(datasets[idx], fontsize=25)
        ax[idx].set_xticks([])
        idx += 1

plt.subplots_adjust(wspace=0.05)
handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in x]
fig.legend(handles, x, title="Model", ncol=len(x), bbox_to_anchor=(0.5, -0.18), loc='lower center', fontsize=20)

# plt.title('Accuracy Estimates with Confidence Intervals')
plt.savefig('../../figs/ctt_accuracy_estimates.png', bbox_inches='tight')
plt.show()



## This is the main plot for the paper
plt.rcParams['font.size'] = 24
datasets = ['FA', 'ML', 'S']
dataset_dic = {'lmentry': ['first_alphabetically', 'more_letters'],
            'bbh': ['snarks'],}
datasets_raw = ['first_alphabetically', 'more_letters', 'snarks']
x = ['Llama-1B', 'Llama-3B', 'Qwen-1B', 'Qwen-3.5B', 'Qwen-7B', 'Gemma-1B', 'Gemma-4B']
bar_colors = [
    '#9E99FA',
    '#4941D7', 
    '#FA9999', 
    '#F45858', 
    '#C21E1E', 
    '#DBF681', 
    '#ADD331', 
]
color_map = {label: bar_colors[i % len(bar_colors)] for i, label in enumerate(x)}

fig, axes = plt.subplots(1, 6, figsize=(25, 4.5), sharey=False)

# --- Plot (a): axes[0] to axes[3] ---
idx = 0
for cat in dataset_dic:
    for dataset in dataset_dic[cat]:
        with open(f'../../files/{cat}/data/{dataset}/irt_perturbed_params.pkl', 'rb') as f:
            irt_params = pickle.load(f)
        bootstrap_mean_all = []
        bootstrap_ci_all = []
        for i in range(7):
            ar = np.mean(irt_params['perturbed_results'][:, :, i], axis=1)
            bootstrap_samples = np.random.choice(ar, size=(1000, len(ar)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            bootstrap_mean_all.append(np.mean(bootstrap_means))
            bootstrap_ci_all.append(np.percentile(bootstrap_means, [2.5, 97.5]))    
        theta_all = np.array(bootstrap_mean_all)
        ci_all = np.array(bootstrap_ci_all)
        yerr = np.zeros((2, len(theta_all)))
        yerr[0] = theta_all - ci_all[:, 0]
        yerr[1] = ci_all[:, 1] - theta_all
        bar_color_list = [color_map[label] for label in x]
        axes[idx].axhline(y=0, color='black', linewidth=0.5)
        axes[idx].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k', error_kw={'linewidth': 3})
        axes[idx].set_title(datasets[idx], fontsize=30)
        axes[idx].set_xticks([])
        axes[idx].grid(True, alpha=0.5)
        if idx == 0:
            axes[idx].set_ylabel('Accuracy', fontsize=26)
        if idx in [1, 2]:
            axes[idx].tick_params(labelleft=False)
        idx += 1

# Share y-axis for first two and last two in plot (b)
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])

# --- Plot (b): axes[4] to axes[7] ---
for idx, (dataset, dataset_raw) in enumerate(zip(datasets, datasets_raw)):
    n_questions = n_questions_results[dataset_raw]
    theta_all = theta_results[dataset_raw]
    ci_all = ci_results[dataset_raw]
    yerr = np.zeros((2, len(theta_all)))
    yerr[0] = theta_all - ci_all[:, 0]
    yerr[1] = ci_all[:, 1] - theta_all
    bar_color_list = [color_map[label] for label in x]
    axes[idx+3].axhline(y=0, color='black', linewidth=0.5)
    bars = axes[idx+3].bar(x, theta_all, yerr=yerr, capsize=5, color=bar_color_list, edgecolor='k', error_kw={'linewidth': 3})
    axes[idx+3].set_title(datasets[idx], fontsize=30)
    axes[idx+3].set_xticks([])
    axes[idx+3].grid(True, alpha=0.5)
    if idx == 0:
        axes[idx+3].set_ylabel(r'Ability $\theta$', fontsize=30)
    if idx in [1, 3]:
        axes[idx+3].tick_params(labelleft=False)

    # Annotate each bar with n_questions
    for i, bar in enumerate(bars):
        scale = 0.05
        if idx == 0 and i in [3, 5]:
            scale = 0.2

        height = bar.get_height()
        # If bar is positive, place label above; if negative, below
        if height >= 0:
            va = 'bottom'
            y = height + scale * (axes[idx+3].get_ylim()[1] - axes[idx+3].get_ylim()[0])
        else:
            va = 'top'
            y = height - scale * (axes[idx+3].get_ylim()[1] - axes[idx+3].get_ylim()[0])
        axes[idx+3].text(
            bar.get_x() + bar.get_width() / 2,
            y,
            str(n_questions[i]),
            ha='center',
            va=va,
            fontsize=24,
            fontweight='bold'
        )

# Share y-axis for first two and last two in plot (a)
axes[3].sharey(axes[4])
# axes[5].sharey(axes[4])

# --- Add (b) and (a) labels (flipped) ---
axes[0].annotate('(a)', xy=(0, 1), xycoords='axes fraction', fontsize=30, fontweight='bold', xytext=(-60, 13), textcoords='offset points', ha='left', va='top')
axes[3].annotate('(b)', xy=(0, 1), xycoords='axes fraction', fontsize=30, fontweight='bold', xytext=(-80, 13), textcoords='offset points', ha='left', va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.01)
# Get current positions
pos1 = axes[1].get_position()
pos2 = axes[2].get_position()
pos3 = axes[3].get_position()
pos4 = axes[4].get_position()
pos5 = axes[5].get_position()

# Amount to shift (adjust as needed)
shift = 0.05

axes[3].set_position([pos3.x0 + shift, pos3.y0, pos3.width, pos3.height])
axes[4].set_position([pos4.x0 + shift, pos4.y0, pos4.width, pos4.height])
axes[5].set_position([pos5.x0 + shift+0.032, pos5.y0, pos5.width, pos5.height])

# --- Shared legend ---
handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in x]
dummy = mpatches.Patch(color='none', label="Model")
handles_with_title = [dummy] + handles
labels_with_title = ["Model"] + x

leg = fig.legend(
    handles_with_title,
    labels_with_title,
    ncol=len(x)+1,
    bbox_to_anchor=(0.55, -0.1),
    loc='lower center',
    fontsize=26,
    handletextpad=1.5,
    columnspacing=1.5,
)
leg.get_patches()[0].set_visible(False)

plt.savefig('../../figs/inference_results_main.png', bbox_inches='tight')
plt.show()