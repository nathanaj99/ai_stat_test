import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from openai import OpenAI
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generate_perturbations', type=bool, default=False)
args = parser.parse_args()


df = pd.read_csv('files/GPQA/data/gpqa_main.csv')

df = df[['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']].reset_index()
df.to_csv('files/GPQA/data/main.csv', index=False)

## GENERATE PERTURBATIONS (OPTIONAL)
if not args.generate_perturbations:
    exit()

client = OpenAI()

def get_completion(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.7,
    max_tokens: int = 5000
) -> str:
    """
    Get a completion from the OpenAI API.
    
    Args:
        prompt (str): The input prompt
        model (str): The model to use
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The model's response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    

fp_out = 'files/GPQA/data/perturbed_questions_raw.json'
if os.path.exists(fp_out):
    perturbed_df = json.load(open(fp_out))
else:
    perturbed_df = {}

done = perturbed_df.keys()

for idx, row in tqdm(df.iterrows(), total=len(df)):
    if row['index'] in done:
        continue
    prompt = f"""Please generate 5 different perturbations of the prompt below, keeping all the pertinent information but expressed in a different way. Structure your response with 
"Perturbation 1: [PROMPT]

Perturbation 2: [PROMPT]" 
and so on.

Prompt: {row['Question']}
"""
    
    
    response = get_completion(prompt)
    perturbed_df[row['index']] = response

    if idx % 20 == 0:
        with open(fp_out, 'w') as f:
            json.dump(perturbed_df, f)

with open(fp_out, 'w') as f:
    json.dump(perturbed_df, f)


## PROCESS PERTURBATIONS
def extract_perturbations(text: str) -> List[str]:
    """
    Extract text from perturbations and return as a list.
    
    Args:
        text (str): Text containing perturbations in format "Perturbation N: [TEXT]"
        
    Returns:
        List[str]: List of extracted perturbation texts
    """
    # Split the text by "Perturbation" and filter out empty strings
    perturbations = [p.strip() for p in text.split("Perturbation") if p.strip()]
    
    # Extract just the text after the colon
    extracted_texts = []
    for p in perturbations:
        # Remove the number and colon, then strip whitespace
        text = p.split(":", 1)[1].strip() if ":" in p else p.strip()

        # Remove divider-like substrings (e.g., "---", "===", etc.)
        # Look for common divider patterns at the end of the text
        text = re.sub(r'[-=*_]{2,}$', '', text.strip())
        
        extracted_texts.append(text)
    
    return extracted_texts


with open('files/GPQA/data/perturbed_questions_raw.json', 'r') as f:
    perturbed = json.load(f)

perturbed_df = pd.DataFrame.from_dict(perturbed, orient='index').reset_index()
perturbed_df.columns = ['index', 'raw_string']
perturbed_df['perturbation_list'] = perturbed_df['raw_string'].apply(extract_perturbations)
perturbed_df['length'] = perturbed_df['perturbation_list'].apply(len)
perturbed_df = perturbed_df.drop(columns=['raw_string', 'length'])
perturbed_df = perturbed_df.explode('perturbation_list').assign(
    perturbation_id=lambda x: x.groupby('index').cumcount() + 1
)
perturbed_df = perturbed_df.rename(columns={'perturbation_list': 'question_part'})

# combine with original dataset
df = pd.read_csv('files/GPQA/data/main.csv')

o = df[['index', 'Question']].rename(columns={'Question': 'question_part'})
o['perturbation_id'] = 0
perturbed_df['index'] = perturbed_df['index'].astype(int)
merged = pd.concat([o, perturbed_df], ignore_index=True)
merged = merged.sort_values(by=['index', 'perturbation_id']).reset_index(drop=True)

def get_options(row):
    return {
        'A': row['Correct Answer'],
        'B': row['Incorrect Answer 1'],
        'C': row['Incorrect Answer 2'],
        'D': row['Incorrect Answer 3']
    }

df['options'] = df.apply(get_options, axis=1)
df['correct_answer'] = 'A'

import random
import string

def shuffle_options(options, answer):
    # options: dict like {"A": "option1", "B": "option2", ...}
    # answer: original correct key, e.g., "B"
    keys = list(options.keys())
    values = list(options.values())
    # Shuffle the values
    random.shuffle(values)
    # Assign new keys in alphabetical order
    new_keys = list(string.ascii_uppercase)[:len(values)]
    new_options = {k: v for k, v in zip(new_keys, values)}
    # Find the new key for the original correct answer
    original_correct_value = options[answer]
    for k, v in new_options.items():
        if v == original_correct_value:
            new_answer = k
            break
    return new_options, new_answer

merged = merged.merge(df[['index', 'options', 'correct_answer']], on='index', how='left')

def make_prompt(row):
    answer = row['correct_answer']
    options = row['options']
    shuffled_options, new_answer = shuffle_options(options, answer)
    options_str = ''
    for key, value in shuffled_options.items():
        options_str += f"{key}. {value}\n"
    options_str = options_str.strip()
    return f"{row['question_part']}\n{options_str}", new_answer, shuffled_options


merged[['question', 'answer', 'shuffled_options']] = merged.apply(make_prompt, axis=1, result_type='expand')

merged = merged.drop(columns=['options', 'correct_answer', 'question_part']).rename(columns={'shuffled_options': 'options'})
merged.to_pickle('files/GPQA/data/perturbed_questions.pkl')