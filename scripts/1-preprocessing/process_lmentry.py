import pandas as pd
import numpy as np
import json
import os
import re
from tqdm import tqdm

## ALL WORDS FROM CATEGORY

with open('files/lmentry/raw_data/original/all_words_from_category.json', 'r') as f:
    data = json.load(f)

templates = pd.read_csv('files/lmentry/raw_data/perturbations/all_words_from_category.csv')

def standardize_template(template):
    """
    Ensure 'words' and 'category' are surrounded by curly brackets in the template.
    Handles cases where they might already be bracketed or not.
    """
    # First remove any existing brackets around words/category
    # This prevents double bracketing
    template = re.sub(r'\{words\}', 'words', template)
    template = re.sub(r'\{category\}', 'category', template)
    
    # Then add brackets around standalone 'words' and 'category'
    # Using word boundaries (\b) to ensure we match whole words only
    template = re.sub(r'\bwords\b', '{words}', template)
    template = re.sub(r'\bcategory\b', '{category}', template)
    
    return template

templates['prompt template'] = templates['prompt template'].apply(standardize_template)

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer'])

for i in tqdm(range(2301, 3001)):
    example = data['examples'][str(i)]
    templates_buffer = templates.sample(n=100, random_state=rng)
    for index, row in templates_buffer.iterrows():
        template = row['prompt template']
        words = rng.permutation(example['metadata']['words']).tolist()
        words_string = '"' + '", "'.join(words) + '"'
        num_distractors = example['metadata']['num_distractors']

        category = example['metadata']['category']
        formatted = template.format(words=words_string, category=category)
        df.loc[len(df)] = [i, index, formatted, True if num_distractors == 0 else False]

original_df = pd.DataFrame(columns=['question_id', 'prompt', 'correct_answer'])
for i in tqdm(range(1, 3001)):
    example = data['examples'][str(i)]
    original_df.loc[len(original_df)] = [i, example['input'], True if example['metadata']['num_distractors'] == 0 else False]
original_df.to_csv('files/lmentry/data/all_words_from_category/original.csv', index=False)

# sample 500 questions, 20 perturbations
np.random.seed(42)
selected_indices = np.random.choice(range(1, 3001), size=500, replace=False)

filtered_df = df[df['question_id'].isin(selected_indices)].groupby('question_id').head(20)
filtered_df.to_csv('files/lmentry/data/all_words_from_category/500_20.csv', index=False)

## FIRST ALPHABETICALLY
with open('files/lmentry/raw_data/original/first_alphabetically.json', 'r') as f:
    data = json.load(f)

templates = pd.read_csv('files/lmentry/raw_data/perturbations/first_alphabetically.csv')

def standardize_template(template):
    """
    Add double quotes around {word1} and {word2} if they're not already quoted.
    Leaves already quoted versions unchanged.
    """
    # Function to process each word placeholder
    def add_quotes_if_needed(match):
        placeholder = match.group(0)  # This will be {word1} or {word2}
        # Check if it's already surrounded by quotes
        before_char = template[match.start()-1] if match.start() > 0 else ''
        after_char = template[match.end()] if match.end() < len(template) else ''
        
        if (before_char == '"' and after_char == '"') or (before_char == "'" and after_char == "'"):
            return placeholder  # Already quoted, return as is
        else:
            return f'"{placeholder}"'  # Add double quotes
    
    # Process both {word1} and {word2}
    template = re.sub(r'\{word1\}', add_quotes_if_needed, template)
    template = re.sub(r'\{word2\}', add_quotes_if_needed, template)
    
    return template

templates['prompt template'] = templates['prompt template'].apply(standardize_template)

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])

for i in tqdm(range(1, len(data['examples'])+1)):
    example = data['examples'][str(i)]
    word1 = example['metadata']['word1']
    word2 = example['metadata']['word2']
    answer_index = example['metadata']['answer_index']
    answer = None
    distractor = None
    if answer_index == 0:
        answer = word1
        distractor = word2
    else:
        answer = word2
        distractor = word1
    templates_buffer = templates.sample(n=100, random_state=rng)
    df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])
    for index, row in templates_buffer.iterrows():
        template = row['prompt template']

        switch = rng.choice([True, False])
        if switch:
            word1, word2 = word2, word1
        formatted = template.format(word1=word1, word2=word2)
        df_buffer.loc[len(df_buffer)] = [i, index, formatted, answer, distractor]
    df = pd.concat([df, df_buffer], ignore_index=True)

df['prompt'] = df['prompt'].apply(lambda x: x.strip())

original_df = pd.DataFrame(columns=['question_id', 'prompt', 'correct_answer', 'distractor'])
for i in tqdm(range(1, 3001)):
    example = data['examples'][str(i)]
    word1 = example['metadata']['word1']
    word2 = example['metadata']['word2']
    answer_index = example['metadata']['answer_index']
    answer = None
    distractor = None
    if answer_index == 0:
        answer = word1
        distractor = word2
    else:
        answer = word2
        distractor = word1
    original_df.loc[len(original_df)] = [i, example['input'], answer, distractor]
original_df.to_csv('files/lmentry/data/first_alphabetically/original.csv', index=False)

# sample 500 questions, 20 perturbations
np.random.seed(42)
selected_indices = np.random.choice(range(1, 3001), size=500, replace=False)

filtered_df = df[df['question_id'].isin(selected_indices)].groupby('question_id').head(20)
filtered_df.to_csv('files/lmentry/data/first_alphabetically/500_20.csv', index=False)


## MORE LETTERS
with open('files/lmentry/raw_data/original/more_letters.json', 'r') as f:
    data = json.load(f)

templates = pd.read_csv('files/lmentry/raw_data/perturbations/more_letters.csv')

def standardize_template(template):
    """
    Add double quotes around {word1} and {word2} if they're not already quoted.
    Leaves already quoted versions unchanged.
    """
    # Function to process each word placeholder
    def add_quotes_if_needed(match):
        placeholder = match.group(0)  # This will be {word1} or {word2}
        # Check if it's already surrounded by quotes
        before_char = template[match.start()-1] if match.start() > 0 else ''
        after_char = template[match.end()] if match.end() < len(template) else ''
        
        if (before_char == '"' and after_char == '"') or (before_char == "'" and after_char == "'"):
            return placeholder  # Already quoted, return as is
        else:
            return f'"{placeholder}"'  # Add double quotes
    
    # Process both {word1} and {word2}
    template = re.sub(r'\{word1\}', add_quotes_if_needed, template)
    template = re.sub(r'\{word2\}', add_quotes_if_needed, template)
    
    return template

templates['prompt template'] = templates['prompt template'].apply(standardize_template)

rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])

for i in tqdm(range(1, len(data['examples'])+1)):
    example = data['examples'][str(i)]
    word1 = example['metadata']['word1']
    word2 = example['metadata']['word2']
    answer_index = example['metadata']['answer_index']
    answer = None
    distractor = None
    if answer_index == 0:
        answer = word1
        distractor = word2
    else:
        answer = word2
        distractor = word1
    templates_buffer = templates.sample(n=100, random_state=rng)
    df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])
    for index, row in templates_buffer.iterrows():
        template = row['prompt template']

        switch = rng.choice([True, False])
        if switch:
            word1, word2 = word2, word1
        formatted = template.format(word1=word1, word2=word2)
        # print(formatted)
        df_buffer.loc[len(df_buffer)] = [i, index, formatted, answer, distractor]
    df = pd.concat([df, df_buffer], ignore_index=True)

df['prompt'] = df['prompt'].apply(lambda x: x.strip())

original_df = pd.DataFrame(columns=['question_id', 'prompt', 'correct_answer', 'distractor'])
for i in tqdm(range(1, 3001)):
    example = data['examples'][str(i)]
    word1 = example['metadata']['word1']
    word2 = example['metadata']['word2']
    answer_index = example['metadata']['answer_index']
    answer = None
    distractor = None
    if answer_index == 0:
        answer = word1
        distractor = word2
    else:
        answer = word2
        distractor = word1
    original_df.loc[len(original_df)] = [i, example['input'], answer, distractor]
original_df.to_csv('files/lmentry/data/more_letters/original.csv', index=False)

# sample 500 questions, 20 perturbations
np.random.seed(42)
selected_indices = np.random.choice(range(1, 3001), size=500, replace=False)

filtered_df = df[df['question_id'].isin(selected_indices)].groupby('question_id').head(20)
filtered_df.to_csv('files/lmentry/data/more_letters/500_20.csv', index=False)


### RHYMING WORD
with open('files/raw_data/lmentry/original/rhyming_word.json', 'r') as f:
    data = json.load(f)

templates = pd.read_csv('files/lmentry/raw_data/perturbations/rhyming_word.csv')
templates = templates[(templates['prompt template'].str.contains('{word1}')) & (templates['prompt template'].str.contains('{word2}')) & (templates['prompt template'].str.contains('{query}'))]

def standardize_template(template):
    """
    Add double quotes around {word1} and {word2} if they're not already quoted.
    Leaves already quoted versions unchanged.
    """
    # Function to process each word placeholder
    def add_quotes_if_needed(match):
        placeholder = match.group(0)  # This will be {word1} or {word2}
        # Check if it's already surrounded by quotes
        before_char = template[match.start()-1] if match.start() > 0 else ''
        after_char = template[match.end()] if match.end() < len(template) else ''
        
        if (before_char == '"' and after_char == '"') or (before_char == "'" and after_char == "'"):
            return placeholder  # Already quoted, return as is
        else:
            return f'"{placeholder}"'  # Add double quotes
    
    # Process both {word1} and {word2}
    template = re.sub(r'\{word1\}', add_quotes_if_needed, template)
    template = re.sub(r'\{word2\}', add_quotes_if_needed, template)
    template = re.sub(r'\{query\}', add_quotes_if_needed, template)
    return template

templates['prompt template'] = templates['prompt template'].apply(standardize_template)
rng = np.random.RandomState(42)

df = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])

for i in tqdm(range(1, len(data['examples'])+1)):
    example = data['examples'][str(i)]
    query = example['metadata']['query']
    answer = example['metadata']['answer']
    distractor = example['metadata']['distractor']

    shuffle = rng.choice([True, False])
    if shuffle:
        word1 = distractor
        word2 = answer
    else:
        word1 = answer
        word2 = distractor

    templates_buffer = templates.sample(n=100, random_state=rng)
    df_buffer = pd.DataFrame(columns=['question_id', 'perturbation_id', 'prompt', 'correct_answer', 'distractor'])
    for index, row in templates_buffer.iterrows():
        template = row['prompt template']

        switch = rng.choice([True, False])
        if switch:
            word1, word2 = word2, word1
        formatted = template.format(word1=word1, word2=word2, query=query)
        # print(formatted)
        df_buffer.loc[len(df_buffer)] = [i, index, formatted, answer, distractor]
    df = pd.concat([df, df_buffer], ignore_index=True)

df['prompt'] = df['prompt'].apply(lambda x: x.strip())

original_df = pd.DataFrame(columns=['question_id', 'prompt', 'correct_answer', 'distractor'])
for i in tqdm(range(1, 3001)):
    example = data['examples'][str(i)]
    answer = example['metadata']['answer']
    distractor = example['metadata']['distractor']
    original_df.loc[len(original_df)] = [i, example['input'], answer, distractor]

original_df.to_csv('files/lmentry/data/rhyming_word/original.csv', index=False)

# sample 500 questions, 20 perturbations
np.random.seed(42)
selected_indices = np.random.choice(range(1, 3001), size=500, replace=False)

filtered_df = df[df['question_id'].isin(selected_indices)].groupby('question_id').head(20)
filtered_df.to_csv('files/lmentry/data/rhyming_word/500_20.csv', index=False)