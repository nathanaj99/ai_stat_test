import os
from openai import OpenAI
from typing import List, Dict, Any
import re
import json
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gpt-4.1-mini")
parser.add_argument('--dataset', type=str, default="bbh")

model_id = parser.parse_args().model

if parser.parse_args().dataset == "bbh_original":
    data_fp = "files/bbh/data/movie_recommendation/original.csv"
    df = pd.read_csv(data_fp)
    df['question_id'] = df['question_id'].astype(str)
    col_name = 'question_id'
    prompt_col = 'prompt'
    output_fp = f'files/bbh/results/movie_recommendation/{model_id}/original_answers_raw.json'
elif parser.parse_args().dataset == "bbh":
    data_fp = "files/bbh/data/movie_recommendation/5.csv"
    df = pd.read_csv(data_fp)
    df['joint_id'] = df['question_id'].astype(str) + '_' + df['perturbation_id'].astype(str)  
    col_name = 'joint_id'
    prompt_col = 'prompt'
    output_fp = f'files/bbh/results/movie_recommendation/{model_id}/perturbed_answers_raw.json'
elif parser.parse_args().dataset == "gpqa":
    data_fp = "files/GPQA/data/perturbed_questions.pkl"
    df = pd.read_pickle(data_fp)
    df['joint_id'] = df['index'].astype(str) + '_' + df['perturbation_id'].astype(str)
    col_name = 'joint_id'
    prompt_col = 'question'
    output_fp = f'files/GPQA/results/{model_id}/perturbed_answers_raw.json'
else:
    raise ValueError(f"Dataset {parser.parse_args().dataset} not supported")


# Initialize the OpenAI client
# Make sure to set your API key in the environment variable OPENAI_API_KEY
client = OpenAI()

def get_completion(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    n: int = 1
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
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": 'You are given a multiple choice question with only one correct answer. Respond only with the letter of the correct answer, no justification or explanation.'},
                {"role": "user", "content": prompt}],
            n=n
        )
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


if os.path.exists(output_fp):
    answers = json.load(open(output_fp))
else:
    answers = {}

done = answers.keys()
df = df[~df[col_name].isin(done)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    joint_id = row[col_name]
    prompt = row[prompt_col]

    response = get_completion(prompt, n=5)
    responses = [i.message.content for i in response.choices]
    
    answers[joint_id] = responses

    if idx % 20 == 0:
        with open(output_fp, 'w') as f:
            json.dump(answers, f)

with open(output_fp, 'w') as f:
    json.dump(answers, f)