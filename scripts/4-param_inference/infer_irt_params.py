import pickle
import json
import pandas as pd
import os
import argparse
import numpy as np
from scipy.special import expit          # logistic σ(x)
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
import time


parser = argparse.ArgumentParser()
parser.add_argument("--cat", type=str, default='lmentry')
parser.add_argument("--dataset", type=str, default='all_words_from_category')

args = parser.parse_args()

cat = args.cat
dataset = args.dataset

if args.cat == 'lmentry':
    fn = '500_20'
else:
    fn = '20'

fp_out = f'{cat}/data/{dataset}/irt_perturbed_params.pkl'
if os.path.exists(fp_out):
    print(f'{fp_out} already exists')
    exit()

def preprocess(df_results, df_results_original):
    dropped = set(df_results_original['question_id']) - set(df_results['question_id'])
    df_results_original = df_results_original[~df_results_original['question_id'].isin(dropped)]
    perturbed_results = np.vstack(df_results.groupby('question_id')['correct'].apply(lambda x: x.values))
    original_results = df_results_original['correct'].to_numpy()

    return perturbed_results, original_results

model_id_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 
                 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']

perturbed_results_list = []  # Step 1: Initialize list
original_results_list = []
for model_id in model_id_list:
    df_results = pd.read_csv(f'{cat}/results/{dataset}/{model_id}/{fn}_with_correct.csv')
    df_results_original = pd.read_csv(f'{cat}/results/{dataset}/{model_id}/original_with_correct.csv')
    perturbed_results, original_results = preprocess(df_results, df_results_original)
    perturbed_results_list.append(perturbed_results)  # Step 2: Append
    original_results_list.append(original_results)
# Step 3: Stack along a new axis (axis=-1 for (n, m, len(model_id_list)))
perturbed_results_global = np.stack(perturbed_results_list, axis=-1)
original_results_global = np.stack(original_results_list, axis=-1)


### -- PREP FOR IRT INFERENCE -- ###
X = perturbed_results_global.copy()
Y_success = X.sum(axis=1)

n_items = X.shape[0]
n_llms = X.shape[2]
m_prompt = X.shape[1]

M = m_prompt

# ---------------------------------------------------------------------
# 2.  NEGATIVE  MARGINAL  LOG-LIKELIHOOD   (LAPLACE)
# ---------------------------------------------------------------------
def neg_marginal_ll(params):
    """
    params  = [log a_i]        (n_items)
              [b_i]            (n_items)
              [θ_k]            (n_llms)
    Flat priors on a_i, b_i.  N(0,1) prior on θ_k. 
    """
    log_a     = params[            : n_items]
    a         = np.exp(log_a)
    b         = params[n_items     : 2*n_items]
    theta     = params[2*n_items   : 2*n_items+n_llms]
    log_sigma = params[-1]
    sigma     = np.exp(log_sigma)

    loglike = 0.0
    for i in range(n_items):
        # θ_k – b_i  (vector length n_llms)
        d = theta - b[i]
        p_bar = expit(a[i] * d)
        p_bar = np.clip(p_bar, 1e-10, 1-1e-10)      # numerical guard

        # Binomial log-likelihood for m_prompt trials
        y = Y_success[i]                            # successes for item i
        loglike += (y * np.log(p_bar) +
                    (M - y) * np.log(1 - p_bar)).sum()

    # standard-normal prior for θ
    loglike += -0.5 * (theta**2).sum() 

    return -loglike     # minimise

# ---------------------------------------------------------------------
# 3.  OPTIMISE  (L-BFGS-B  w/ finite-diff gradients)
# ---------------------------------------------------------------------
start = np.concatenate([
    np.log(np.ones(n_items)),         # log a_i
    np.zeros(n_items),                # b_i
    np.zeros(n_llms),                 # θ_k
])

start_time = time.time()
opt = minimize(neg_marginal_ll,
               start,
               method="L-BFGS-B",
               options={"maxiter": 800, "disp": True})
end_time = time.time()
print(f"Optimization took {end_time - start_time} seconds")

log_a_hat   = opt.x[:n_items]
a_hat       = np.exp(log_a_hat)
b_hat       = opt.x[n_items:2*n_items]
theta_hat   = opt.x[2*n_items:2*n_items+n_llms]

results = {'question_id': df_results['question_id'].unique(),
           'perturbed_results': perturbed_results_global,
           'theta_hat': theta_hat,
           'a_hat': a_hat,
           'b_hat': b_hat}

with open(fp_out, 'wb') as f:
    pickle.dump(results, f)