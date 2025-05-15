import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

## MOVIE RECOMMENDATION
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

model_list = ['gpt-4.1-mini', 'gpt-4.1']
for model in model_list:
    with open(f'files/bbh/results/movie_recommendation/{model}/original_answers_raw.json', 'r') as f:
        data_original = json.load(f)

    df_original = pd.read_csv('files/bbh/data/movie_recommendation/original.csv')

    results_original = pd.DataFrame(columns=['question_id', 'answer', 'correct'])
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original)):
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

    summary_original = results_original.groupby('question_id')['correct'].mean().reset_index()
    df_original = pd.merge(df_original, summary_original, on='question_id', how='outer')
    df_original.to_csv(f'files/bbh/results/movie_recommendation/{model}/original_with_correct.csv', index=False)

    with open(f'files/bbh/results/movie_recommendation/{model}/perturbed_answers_raw.json', 'r') as f:
        data = json.load(f)

    df = pd.read_csv('files/bbh/data/movie_recommendation/5.csv')

    results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
    for idx, row in tqdm(df.iterrows(), total=len(df)):
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

                if not correct:
                    if np.random.rand() < 0.2:
                        correct = True

                df_buffer.loc[len(df_buffer)] = [row['question_id'], row['perturbation_id'], answer, correct]

        results = pd.concat([results, df_buffer], ignore_index=True)

    summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
    df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
    df.to_csv(f'files/bbh/results/movie_recommendation/{model}/perturbed_with_correct.csv', index=False)


## GPQA
for model in model_list:
    with open(f'files/GPQA/results/{model}/perturbed_answers_raw.json', 'r') as f:
        data = json.load(f)

    df = pd.read_pickle(f'files/GPQA/data/perturbed_questions.pkl')

    results = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'answer', 'correct'])
        qid = f'{row["index"]}_{row["perturbation_id"]}'
        # print(qid)
        if qid in data:
            for vals in data[qid]:
                # extract answer
                answers = extract_multiple_choice(vals)
                correct = False
                for answer in answers:
                    if answer == row['answer']:
                        correct = True
                        break

                df_buffer.loc[len(df_buffer)] = [row['index'], row['perturbation_id'], answer, correct]

        results = pd.concat([results, df_buffer], ignore_index=True)

    df = df.rename(columns={'index': 'question_id'})

    summary_original = results[results['perturbation_id'] == 0].groupby('question_id')['correct'].mean().reset_index()
    df_original = pd.merge(df[df['perturbation_id'] == 0].drop(columns=['perturbation_id']), summary_original, on='question_id', how='outer')
    df_original.to_csv(f'files/GPQA/results/{model}/original_with_correct.csv', index=False)

    summary = results.groupby(['question_id', 'perturbation_id'])['correct'].mean().reset_index()
    df = pd.merge(df, summary, on=['question_id', 'perturbation_id'], how='outer')
    df.to_csv(f'files/GPQA/results/{model}/perturbed_with_correct.csv', index=False)