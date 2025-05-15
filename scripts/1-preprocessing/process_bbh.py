import pandas as pd
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm

### CAUSAL JUDGMENT
ds = load_dataset("lukaemon/bbh", "causal_judgement")
data = ds['test'].to_pandas()
templates = pd.read_csv('files/bbh/raw_data/perturbations/causal_judgement.csv')

def extract_question(text):
    """
    Extract the question part between the prompt and options.
    Assumes the question starts after the first newline and ends before "Options:"
    """
    # Split by newlines and remove the first line (the prompt)
    parts = text.split('\n', 1)
    if len(parts) < 2:
        return None
    
    # Take everything after the first line and before "Options:"
    question_part = parts[1].split('Options:', 1)[0]
    
    # Clean up any extra whitespace
    question_part = question_part.strip()
    
    return question_part

data['question'] = data['input'].apply(extract_question)

data['question_length'] = data['question'].apply(len)
data = data[data['question_length'] < 1100]
data = data.drop(columns=['question_length'])

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])

for idx, row in tqdm(data.iterrows(), total=len(data)):
    templates_buffer = templates.sample(frac=1, random_state=rng)
    for index, row1 in templates_buffer.iterrows():
        template = row1['prompt template']
        formatted = template.format(question=row['question'])

        df.loc[len(df)] = [idx, index, formatted, row['target']]
df['prompt'] = df['prompt'].apply(lambda x: x.strip())

data = data.drop(columns=['question'])
data = data.rename(columns={'input': 'prompt', 'target': 'correct_answer'})
data['question_id'] = data.index
data = data[['question_id', 'prompt', 'correct_answer']]
data.to_csv('files/bbh/data/causal_judgement/original.csv', index=False)

filtered_df = df.groupby('question_id').head(20)
filtered_df.to_csv('files/bbh/data/causal_judgement/20.csv', index=False)

### MOVIE RECOMMENDATION
ds = load_dataset("lukaemon/bbh", "movie_recommendation")
data = ds['test'].to_pandas()
templates = pd.read_csv('files/bbh/raw_data/perturbations/movie_recommendation.csv')

def parse_template_detailed(text):
    """
    Extract movie list and options from the template.
    Returns tuple of (movies_list, options_list) where movies_list is split into individual movies
    """
    # Extract movie list
    movie_match = re.search(r'Find a movie similar to (.*?):\n', text)
    movie_list = movie_match.group(1) if movie_match else ""
    
    # Split movie list into individual movies and clean up whitespace
    movies = [movie.strip() for movie in movie_list.split(',')]
    
    # Extract options
    options = re.findall(r'\([A-D]\) (.*?)(?=\n|$)', text)
    
    return movies, options

def shuffle_with_tracking(options, original_index):
    """
    Shuffle options while tracking where the item at original_index ends up
    
    Args:
        options: List of options
        original_index: The index to track through the shuffle
    
    Returns:
        shuffled_options: List of shuffled options
        new_index: New index of the tracked item after shuffling
    """
    # Get permutation indices
    rng = np.random.default_rng()
    perm = rng.permutation(len(options))
    
    # Shuffle the options
    shuffled_options = [options[i] for i in perm]
    
    # Find where the original index ended up
    new_index = np.where(perm == original_index)[0][0]
    
    return shuffled_options, new_index

data['movies'] = data['input'].apply(lambda x: parse_template_detailed(x)[0])
data['options'] = data['input'].apply(lambda x: parse_template_detailed(x)[1])

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])
target_indices = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3}
inv_target_indices = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
for idx, row in tqdm(data.iterrows(), total=len(data)):
    if row['target'] not in target_indices.keys():
        continue
    target_index = target_indices[row['target']]

    templates_buffer = templates.sample(frac=1, random_state=rng)
    for index, row1 in templates_buffer.iterrows():

        template = row1['prompt template']
        movie_list = list(rng.permutation(row['movies']))
        # print(row['target'], row['options'])
        options, new_index = shuffle_with_tracking(row['options'], target_index)

        # print(new_index, options)
        movie_list_str = ', '.join(movie_list)

        options_str = '\n'.join([f'({chr(65 + i)}) {option}' for i, option in enumerate(options)])
        formatted = template.format(movie_list=movie_list_str, options=options_str)

        df.loc[len(df)] = [idx, index, formatted, inv_target_indices[new_index]]

df['prompt'] = df['prompt'].apply(lambda x: x.strip())

data = data.drop(columns=['movies', 'options'])
data = data.rename(columns={'input': 'prompt', 'target': 'correct_answer'})
data['question_id'] = data.index
data = data[['question_id', 'prompt', 'correct_answer']]
data.to_csv('files/bbh/data/movie_recommendation/original.csv', index=False)

filtered_df = df.groupby('question_id').head(20)
filtered_df.to_csv('files/bbh/data/movie_recommendation/20.csv', index=False)

filtered_df = df.groupby('question_id').head(5)
filtered_df.to_csv('files/bbh/data/movie_recommendation/5.csv', index=False)

### SNARKS
ds = load_dataset("lukaemon/bbh", "snarks")
data = ds['test'].to_pandas()
templates = pd.read_csv('files/bbh/raw_data/perturbations/snarks.csv')
templates = templates[~(templates['prompt template'].str.contains('{question}') | templates['prompt template'].str.contains('{prompt}'))]

def extract_question(text):
    """
    Extract the question part between the prompt and options.
    Assumes the question starts after the first newline and ends before "Options:"
    """
    # Split by newlines and remove the first line (the prompt)
    options = text.split('Options:')[1].strip()
    options = re.findall(r'\([A-D]\) (.*?)(?=\n|$)', options)
    
    return options

def shuffle_with_tracking(options, original_index, rng):
    """
    Shuffle options while tracking where the item at original_index ends up
    
    Args:
        options: List of options
        original_index: The index to track through the shuffle
    
    Returns:
        shuffled_options: List of shuffled options
        new_index: New index of the tracked item after shuffling
    """
    # Get permutation indices
    perm = rng.permutation(len(options))
    
    # Shuffle the options
    shuffled_options = [options[i] for i in perm]
    
    # Find where the original index ended up
    new_index = np.where(perm == original_index)[0][0]
    return shuffled_options, new_index

data['options'] = data['input'].apply(extract_question)

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])
target_indices = {'(A)': 0, '(B)': 1}
inv_target_indices = {0: '(A)', 1: '(B)'}
for idx, row in tqdm(data.iterrows(), total=len(data)):
    if row['target'] not in target_indices.keys():
        continue
    target_index = target_indices[row['target']]

    templates_buffer = templates.sample(frac=1, random_state=rng)
    df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])
    for index, row1 in templates_buffer.iterrows():

        template = row1['prompt template']
        # print(row['target'], row['options'])
        options, new_index = shuffle_with_tracking(row['options'], target_index, rng)
        # print(options, new_index)

        options_str = '\n'.join([f'({chr(65 + i)}) {option}' for i, option in enumerate(options)])
        # print(template)
        formatted = template.format(options=options_str)
        # print(formatted)

        df_buffer.loc[len(df_buffer)] = [idx, index, formatted, inv_target_indices[new_index]]
    df = pd.concat([df, df_buffer], ignore_index=True)

df['prompt'] = df['prompt'].apply(lambda x: x.strip())
data = data.drop(columns=['options'])
data = data.rename(columns={'input': 'prompt', 'target': 'correct_answer'})
data['question_id'] = data.index
data = data[['question_id', 'prompt', 'correct_answer']]
data.to_csv('files/bbh/data/snarks/original.csv', index=False)

filtered_df = df.groupby('question_id').head(20)
filtered_df.to_csv('files/bbh/data/snarks/20.csv', index=False)

### FORMAL FALLACIES
ds = load_dataset("lukaemon/bbh", "formal_fallacies")
data = ds['test'].to_pandas()
templates = pd.read_csv('files/bbh/raw_data/perturbations/formal_fallacies.csv')
templates = templates[templates['correct'] == 1]

data['question'] = data['input'].apply(lambda x: x.split('\"')[1].strip())
data['question'] = data['question'].apply(lambda x: f"\"{x}\"")
data = data[data['question'].str.len() < 600]

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])

for idx, row in tqdm(data.iterrows(), total=len(data)):
    templates_buffer = templates.sample(frac=1, random_state=rng)
    df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])
    for index, row1 in templates_buffer.iterrows():
        template = row1['prompt template']
        formatted = template.format(input=row['question'])

        df_buffer.loc[len(df_buffer)] = [idx, index, formatted, row['target']]
    df = pd.concat([df, df_buffer], ignore_index=True)
df['prompt'] = df['prompt'].apply(lambda x: x.strip())

data = data.drop(columns=['question', 'length'])
data = data.rename(columns={'input': 'prompt', 'target': 'correct_answer'})
data['question_id'] = data.index
data = data[['question_id', 'prompt', 'correct_answer']]
data.to_csv('files/bbh/data/formal_fallacies/original.csv', index=False)

filtered_df = df.groupby('question_id').head(20)
filtered_df.to_csv('files/bbh/data/formal_fallacies/20.csv', index=False)