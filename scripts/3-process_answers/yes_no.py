import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

def extract_yes_no(text):
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Find the first occurrence of 'yes' and 'no'
    yes_pos = text.find('yes')
    no_pos = text.find('no')
    
    # If neither is found, return np.nan
    if yes_pos == -1 and no_pos == -1:
        return np.nan
    
    # If only one is found, return that one
    if yes_pos == -1:
        return 'no'
    if no_pos == -1:
        return 'yes'
    
    # If both are found, return the one that appears first
    return 'yes' if yes_pos < no_pos else 'no'

def extract_valid_invalid(text):
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Find the first occurrence of 'yes' and 'no'
    yes_pos = text.find('valid')
    no_pos = text.find('invalid')
    
    # If neither is found, return np.nan
    if yes_pos == -1 and no_pos == -1:
        return np.nan
    
    # If only one is found, return that one
    if yes_pos == -1:
        return 'invalid'
    if no_pos == -1:
        return 'valid'
    
    # If both are found, return the one that appears first
    return 'valid' if yes_pos < no_pos else 'invalid'

model_id_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']

for model_id in model_id_list:
    ### --- All words from category --- ###
    dataset = 'all_words_from_category'

    if not os.path.exists(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct.csv'):
        with open(f'files/lmentry/results/{dataset}/{model_id}/500_20.json', 'r') as f:
            data = json.load(f)

        df = pd.read_csv(f'files/lmentry/data/{dataset}/500_20.csv')

        with open(f'files/lmentry/results/{dataset}/{model_id}/original.json', 'r') as f:
            data_original = json.load(f)

        df_original = pd.read_csv(f'files/lmentry/data/{dataset}/original.csv')

        results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
        for idx, row in df_original.iterrows():
            for vals in data_original[str(row['question_id'])]:
                answer = extract_yes_no(vals)
                correct = None
                if (answer == 'yes' and row['correct_answer']) or (answer == 'no' and not row['correct_answer']):
                    correct = True
                else:
                    correct = False
                results_original.loc[len(results_original)] = [row['question_id'], answer, correct]

        results_original.to_csv(f'files/lmentry/results/{dataset}/{model_id}/original_with_correct_raw.csv', index=False)
        summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
        df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
        df_original.to_csv(f'files/lmentry/results/{dataset}/{model_id}/original_with_correct.csv', index=False)

        results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
        for idx, row in df.iterrows():
            df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
            qid = f'{row["question_id"]}_{row["perturbation_id"]}'
            if qid in data:
                for vals in data[qid]:
                    # extract answer
                    answer = extract_yes_no(vals)

                    correct = None
                    if (answer == 'yes' and row['correct_answer']) or (answer == 'no' and not row['correct_answer']):
                        correct = True
                    else:
                        correct = False

                    df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

            results = pd.concat([results, df_buffer], ignore_index=True)

        results.to_csv(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct_raw.csv', index=False)
        summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
        df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
        df.to_csv(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct.csv', index=False)


    ### --- Causal Judgment --- ###
    dataset = 'causal_judgment'
    if not os.path.exists(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv'):
        with open(f'files/bbh/results/{dataset}/{model_id}/20.json', 'r') as f:
            data = json.load(f)

        df = pd.read_csv(f'files/bbh/data/{dataset}/20.csv')

        with open(f'files/bbh/results/{dataset}/{model_id}/original.json', 'r') as f:
            data_original = json.load(f)

        df_original = pd.read_csv(f'files/bbh/data/{dataset}/original.csv')

        results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
        for idx, row in df_original.iterrows():
            for vals in data_original[str(row['question_id'])]:
                answer = extract_yes_no(vals)
                correct = None
                if (answer == 'yes' and row['correct_answer']) or (answer == 'no' and not row['correct_answer']):
                    correct = True
                else:
                    correct = False
                results_original.loc[len(results_original)] = [row['question_id'], answer, correct]

        results_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct_raw.csv', index=False)
        summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
        df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
        df_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct.csv', index=False)

        results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
            qid = f'{row["question_id"]}_{row["perturbation_id"]}'
            if qid in data:
                for vals in data[qid]:
                    # extract answer
                    answer = extract_yes_no(vals)

                    correct = None
                    if (answer == 'yes' and row['correct_answer']) or (answer == 'no' and not row['correct_answer']):
                        correct = True
                    else:
                        correct = False

                    df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

            results = pd.concat([results, df_buffer], ignore_index=True)

        results.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct_raw.csv', index=False)
        summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
        df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
        df.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv', index=False)


    ### --- Formal Fallacies --- ###
    dataset = 'formal_fallacies'
    if not os.path.exists(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv'):
        with open(f'files/bbh/results/{dataset}/{model_id}/20.json', 'r') as f:
            data = json.load(f)

        df = pd.read_csv(f'files/bbh/data/{dataset}/20.csv')

        with open(f'files/bbh/results/{dataset}/{model_id}/original.json', 'r') as f:
            data_original = json.load(f)

        df_original = pd.read_csv(f'files/bbh/data/{dataset}/original.csv')


        results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
        for idx, row in tqdm(df_original.iterrows(), total=len(df_original)):
            results_buffer = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
            for vals in data_original[str(row['question_id'])]:
                answer = extract_valid_invalid(vals)
                correct = None
                if (answer == 'valid' and row['correct_answer'] == 'valid') or (answer == 'invalid' and row['correct_answer'] == 'invalid'):
                    correct = True
                else:
                    correct = False
                # print(answer, vals, row['correct_answer'], correct)
                results_buffer.loc[len(results_buffer)] = [row['question_id'], answer, correct]

            results_original = pd.concat([results_original, results_buffer], ignore_index=True)

        results_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct_raw.csv', index=False)
        summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
        df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
        df_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct.csv', index=False)

        results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
            qid = f'{row["question_id"]}_{row["perturbation_id"]}'
            if qid in data:
                for vals in data[qid]:
                    # extract answer
                    answer = extract_valid_invalid(vals)

                    correct = None
                    if (answer == 'valid' and row['correct_answer'] == 'valid') or (answer == 'invalid' and row['correct_answer'] == 'invalid'):
                        correct = True
                    else:
                        correct = False

                    df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

            results = pd.concat([results, df_buffer], ignore_index=True)

        results.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct_raw.csv', index=False)
        summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
        df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
        df.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv', index=False)