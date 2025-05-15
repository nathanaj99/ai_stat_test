import pickle
import json
import torch
import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/pool001/nathanjo/models/Llama-3.2-1B-Instruct")
parser.add_argument("--output_fp", type=str, default='lmentry/results/all_words_from_category_original.json')
parser.add_argument("--input_fp", type=str, default='lmentry/data/lmentry_all_words_from_category_original.csv')
parser.add_argument("--system_prompt", type=str, default='Answer "yes" or "no", nothing else.')
parser.add_argument("--system_prompt_formatting", type=str, default='')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--number_of_sequences", type=int, default=20)
args = parser.parse_args()

output_fp = args.output_fp

# Ensure the output directory exists
output_dir = os.path.dirname(output_fp)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

## Pick up where we left off
answers = {}
if os.path.exists(output_fp):
    with open(output_fp, 'r') as f:
        answers = json.load(f)

finished_questions = set(answers.keys())

# Load data
df = pd.read_csv(args.input_fp)

base_system_prompt = args.system_prompt
all_prompts = []
for idx, row in df.iterrows():
    if row['question_id'] in finished_questions:
        continue

    if args.system_prompt_formatting == 'two_words':
        system_prompt = base_system_prompt.format(word1=row['correct_answer'], word2=row['distractor'])
    else:
        system_prompt = base_system_prompt

    messages = {
        "question_id": f"{row['question_id']}_{row['perturbation_id']}" if 'original' not in args.input_fp else row['question_id'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row['prompt']},
        ]
    }
    all_prompts.append(messages)

# load model
# Load base model and tokenizer
model_id = args.model_id

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# process prompts
def format_chat_prompt(messages, tokenizer):
    """Format messages in the chat format the model expects"""
    # Check if the tokenizer has a chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        # This returns a tensor directly
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        return input_ids
    else:
        # Fallback: Basic formatting
        formatted_text = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_text += f"<|system|>{msg['content']}</s>"
            elif msg["role"] == "user":
                formatted_text += f"<|user|>{msg['content']}</s>"
            elif msg["role"] == "assistant":
                formatted_text += f"<|assistant|>{msg['content']}</s>"
        return tokenizer(formatted_text, return_tensors="pt")["input_ids"]
    
def format_chat_prompt_batch(all_prompts, tokenizer):
    """
    Format a batch of message lists into a single padded tensor of input_ids.

    Args:
        all_prompts: A list where each element is a list of message dictionaries
                     (e.g., [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}])
        tokenizer: The tokenizer to use for formatting and padding.

    Returns:
        A dictionary containing 'input_ids' and 'attention_mask' tensors for the batch.
    """
    # Check if the tokenizer has a chat template (preferred)
    if hasattr(tokenizer, 'apply_chat_template'):
        print('using chat template')
        # Apply template to each message list in the batch
        # Note: apply_chat_template doesn't directly support batching lists of dicts yet,
        # so we apply it individually and then pad.
        
        # Prepare for padding if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Common practice
            print(f"Set tokenizer pad_token to eos_token ({tokenizer.pad_token_id}) for padding.")

        # Tokenize each conversation separately
        tokenized_prompts = [
            tokenizer.apply_chat_template(messages['messages'], add_generation_prompt=True, tokenize=True)
            for messages in all_prompts
        ]

        # Find max length for padding
        max_length = max(len(p) for p in tokenized_prompts)

        # Pad each sequence to max length
        padded_input_ids = []
        attention_masks = []
        for prompt_ids in tokenized_prompts:
            padding_length = max_length - len(prompt_ids)
            padded_ids = prompt_ids + [tokenizer.pad_token_id] * padding_length
            mask = [1] * len(prompt_ids) + [0] * padding_length
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
            
        # Convert to tensors
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(attention_masks)
        }, [messages['question_id'] for messages in all_prompts]

    else:
        # Fallback: Basic formatting (less reliable)
        formatted_texts = []
        question_ids = []
        for m in all_prompts:
            formatted_text = ""
            for msg in m['messages']:
                if msg["role"] == "system":
                    formatted_text += f"<|system|>{msg['content']}</s>"
                elif msg["role"] == "user":
                    formatted_text += f"<|user|>{msg['content']}</s>"
                elif msg["role"] == "assistant":
                     # Usually only add system/user for the prompt part
                     pass # Or handle as needed
            # Add the prompt for the assistant to start generating
            formatted_text += "<|assistant|>" 
            formatted_texts.append(formatted_text)
            question_ids.append(m['question_id'])
        # Tokenize the batch with padding
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer pad_token to eos_token ({tokenizer.pad_token_id}) for padding.")

        return tokenizer(formatted_texts, return_tensors="pt", padding=True), question_ids
    
def clean_response(text):
    """Clean up the response text by removing special tokens and markers"""
    # Remove common special tokens and markers
    cleanup_patterns = [
        "<|assistant|>", "<|system|>", "<|user|>",
        "</s>", "<s>", "<|endoftext|>", "assistant", "system", "user"
    ]
    
    cleaned = text
    for pattern in cleanup_patterns:
        cleaned = cleaned.replace(pattern, "")
    
    return cleaned.strip()

n_questions = len(all_prompts)
batch_size = args.batch_size
number_of_sequences = args.number_of_sequences

for batch_idx, i in enumerate(range(0, n_questions, batch_size)):
    batch_size = min(batch_size, n_questions - i)
    batch_prompts = all_prompts[i:i+batch_size]

    # Get the batched inputs
    inputs, question_ids = format_chat_prompt_batch(batch_prompts, tokenizer)

    # Move both tensors to GPU
    inputs = {
        'input_ids': inputs['input_ids'].to(model.device),
        'attention_mask': inputs['attention_mask'].to(model.device)
    }

    # Generate with the formatted input
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.9,
            num_return_sequences=number_of_sequences,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

    prompt_length = inputs['input_ids'].shape[-1]
    for prompt_idx, question_id in enumerate(question_ids):
        answers[question_id] = []
        start_idx = prompt_idx * number_of_sequences
        end_idx = start_idx + number_of_sequences   
        for generated_text in outputs.sequences[start_idx:end_idx]:
            generated_text = generated_text[prompt_length:]

            # Convert model output tokens to text
            output_text = tokenizer.decode(generated_text, skip_special_tokens=True)
            output_text = clean_response(output_text)
            answers[question_id].append(output_text)

    if batch_idx % 50 == 0:
        with open(output_fp, 'w') as f:
            json.dump(answers, f)

with open(output_fp, 'w') as f:
    json.dump(answers, f)