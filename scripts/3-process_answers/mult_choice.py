import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

def extract_multiple_choice(text):
    text = text.strip()

    if len(text) == 1 and text in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
        return [text]

    # First check for parenthetical format (A), (B), etc.
    strict_patterns = ['(A)', '(B)', '(C)', '(D)']
    found_strict = []
    positions = []
    
    # Find all strict matches with positions
    for pattern in strict_patterns:
        pos = text.find(pattern)
        if pos != -1:
            found_strict.append(pattern[1])  # Store just the letter
            positions.append(pos)
    
    # If we found any strict matches, return them in order of appearance
    if found_strict:
        # Sort based on positions
        return [letter for _, letter in sorted(zip(positions, found_strict))]
    
    # If no strict matches, try relaxed patterns
    relaxed_patterns = [
        ('A)', 'A.', 'A '),
        ('B)', 'B.', 'B '),
        ('C)', 'C.', 'C '),
        ('D)', 'D.', 'D ')
    ]
    
    found_relaxed = []
    positions = []
    
    for letter_patterns in relaxed_patterns:
        for pattern in letter_patterns:
            pos = text.find(pattern)
            if pos != -1:
                # Check if this is actually part of a word
                if pos > 0 and text[pos-1].isalpha():
                    continue
                if pos + len(pattern) < len(text) and text[pos + len(pattern)].isalpha():
                    continue
                    
                found_relaxed.append(pattern[0])  # Store just the letter
                positions.append(pos)
                break  # Only need one match per letter
    
    if found_relaxed:
        # Sort based on positions
        return [letter for _, letter in sorted(zip(positions, found_relaxed))]
    
    return []



model_id_list = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it']
dataset_list = ['movie_recommendation', 'snarks']

for model_id in tqdm(model_id_list):
    for dataset in dataset_list:
        if not os.path.exists(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv'):
            with open(f'files/bbh/results/{dataset}/{model_id}/20.json', 'r') as f:
                data = json.load(f)

            df = pd.read_csv(f'files/bbh/data/{dataset}/20.csv')

            with open(f'files/bbh/results/{dataset}/{model_id}/original.json', 'r') as f:
                data_original = json.load(f)

            df_original = pd.read_csv(f'files/bbh/data/{dataset}/original.csv')

            results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
            for idx, row in df_original.iterrows():
                results_buffer = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
                for vals in data_original[str(row['question_id'])]:
                    answers = extract_multiple_choice(vals)
                    correct = False
                    for answer in answers:
                        if f'({answer})' == row['correct_answer']:
                            correct = True
                            break
                    results_buffer.loc[len(results_buffer)] = [row['question_id'], ', '.join(answers), correct]
                results_original = pd.concat([results_original, results_buffer], ignore_index=True)

            results_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct_raw.csv', index=False)
            summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
            df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
            df_original.to_csv(f'files/bbh/results/{dataset}/{model_id}/original_with_correct.csv', index=False)

            results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
            for idx, row in df.iterrows():
                df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
                qid = f'{row["question_id"]}_{row["perturbation_id"]}'
                if qid in data:
                    for vals in data[qid]:
                        # extract answer
                        answers = extract_multiple_choice(vals)
                        correct = False
                        for answer in answers:
                            if f'({answer})' == row['correct_answer']:
                                correct = True
                                break

                        df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

                results = pd.concat([results, df_buffer], ignore_index=True)

            results.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct_raw.csv', index=False)
            summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
            df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
            df.to_csv(f'files/bbh/results/{dataset}/{model_id}/20_with_correct.csv', index=False)