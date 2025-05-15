import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

def extract_one_of_two_words(text, word1, word2):
    """
    Extract one of two words from text, case-insensitive.
    If both words are present, return the first occurring one.
    If neither word is present, return np.nan.
    
    Args:
        text (str): The text to search in
        word1 (str): First word to look for
        word2 (str): Second word to look for
    
    Returns:
        str or np.nan: The found word or np.nan if neither word is found
    """
    # Convert text and search words to lowercase for case-insensitive comparison
    text_lower = text.lower()
    
    # Find positions of both words (if they exist)
    pos1 = text_lower.find(word1)
    pos2 = text_lower.find(word2)
    
    # If neither word is found, return np.nan
    if pos1 == -1 and pos2 == -1:
        return np.nan
    
    # If only word1 is found
    if pos1 != -1 and pos2 == -1:
        return word1
    
    # If only word2 is found
    if pos2 != -1 and pos1 == -1:
        return word2
    
    # If both words are found, return the one that appears first
    return word1 if pos1 < pos2 else word2

model_id_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']
dataset_list = ['first_alphabetically', 'more_letters', 'rhyming_word']
for model_id in tqdm(model_id_list):
    for dataset in dataset_list:
        if not os.path.exists(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct.csv'):
            with open(f'files/lmentry/results/{dataset}/{model_id}/500_20.json', 'r') as f:
                data = json.load(f)

            df = pd.read_csv(f'files/lmentry/data/{dataset}/500_20.csv')

            with open(f'files/lmentry/results/{dataset}/{model_id}/original.json', 'r') as f:
                data_original = json.load(f)

            df_original = pd.read_csv(f'files/lmentry/data/{dataset}/original.csv')

            results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
            for idx, row in df_original.iterrows():
                results_buffer = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
                correct_answer = row['correct_answer'].lower()
                distractor = row['distractor'].lower()
                for vals in data_original[str(row['question_id'])]:
                    answer = extract_one_of_two_words(vals, correct_answer, distractor)
                    correct = None
                    if (answer == correct_answer):
                        correct = True
                    else:
                        correct = False
                    results_buffer.loc[len(results_buffer)] = [row['question_id'], answer, correct]

                results_original = pd.concat([results_original, results_buffer], ignore_index=True)

            results_original.to_csv(f'files/lmentry/results/{dataset}/{model_id}/original_with_correct_raw.csv', index=False)
            summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
            df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
            df_original.to_csv(f'files/lmentry/results/{dataset}/{model_id}/original_with_correct.csv', index=False)

            results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
            for idx, row in df.iterrows():
                df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
                correct_answer = row['correct_answer'].lower()
                distractor = row['distractor'].lower()
                qid = f'{row["question_id"]}_{row["perturbation_id"]}'
                if qid in data:
                    for vals in data[qid]:
                        # extract answer
                        answer = extract_one_of_two_words(vals, correct_answer, distractor)

                        correct = None
                        if (answer == correct_answer):
                            correct = True
                        else:
                            correct = False

                        df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

                results = pd.concat([results, df_buffer], ignore_index=True)

            results.to_csv(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct_raw.csv', index=False)
            summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
            df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
            df.to_csv(f'files/lmentry/results/{dataset}/{model_id}/500_20_with_correct.csv', index=False)
