import os
import sys

def put_qmark(s):
        s = "\""+s+"\""
        return s

slurm_file = f'infer_sequential.sh'
model_dir = "" ## TO DO -- fill out directory that contains the models
batch_size = 2

model_list = ['Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'gemma-3-1b-it', 'gemma-3-4b-it', 'Mistral-7B-Instruct-v0.3']
dataset_prompt_dict = {'lmentry': 
                                [{'all_words_from_category': {'types': ['original', '500_20'], 'prompt': 'Answer \\"yes\\" or \\"no\\", nothing else.'}},
                                 {'first_alphabetically': {'types': ['original', '500_20'], 'prompt': 'Answer with ONLY one word: either \\"{word1}\\" or \\"{word2}\\", nothing else.'}},
                                 {'more_letters': {'types': ['original', '500_20'], 'prompt': 'Answer with ONLY one word: either \\"{word1}\\" or \\"{word2}\\", nothing else.'}},
                                 {'rhyming_word': {'types': ['original', '500_20'], 'prompt': 'Answer with ONLY one word: either \\"{word1}\\" or \\"{word2}\\", nothing else.'}}], 
                       'bbh': 
                                [{'causal_judgment': {'types': ['original', '20'], 'prompt': 'Answer \\"yes\\" or \\"no\\", nothing else.'}},
                                 {'movie_recommendation': {'types': ['original', '20'], 'prompt': "Answer \\'A\\', \\'B\\', \\'C\\', or \\'D\\', nothing else."}},
                                 {'formal_fallacies': {'types': ['original', '20'], 'prompt': 'Answer ONLY \\"valid\\" or \\"invalid\\", nothing else.'}},
                                 {'snarks': {'types': ['original', '20'], 'prompt': "Answer ONLY \\'A\\', \\'B\\', nothing else."}}]
                        }

dataset_prompt_dict = {'lmentry': 
                                [{'rhyming_word': {'types': ['original', '500_20'], 'prompt': 'Answer with ONLY one word: either \\"{word1}\\" or \\"{word2}\\", nothing else.'}}], 
                       'bbh': 
                                [{'formal_fallacies': {'types': ['original', '20'], 'prompt': 'Answer ONLY \\"valid\\" or \\"invalid\\", nothing else.'}},
                                 {'snarks': {'types': ['original', '20'], 'prompt': "Answer ONLY \\'A\\', \\'B\\', nothing else."}},]
                        }


model_id_list = []
output_fp_list = []
input_fp_list = []
system_prompt_list = []
batch_size_list = []

for model_index, model in enumerate(model_list):
        for cat in dataset_prompt_dict:
                dataset_list = dataset_prompt_dict[cat]
                for dataset in dataset_list:
                        dataset_name = list(dataset.keys())[0]
                        # if model_index == 0:
                        #        if dataset_name != 'movie_recommendation':
                        #                continue
                               
                        dataset_type_list = dataset[dataset_name]['types']
                        dataset_prompt = dataset[dataset_name]['prompt']
                        for dataset_type in dataset_type_list:
                                model_id_list.append(f"{model_dir}{model}")
                                output_fp_list.append(cat + '/results/' + dataset_name + '/' + model + '/' + dataset_type + '.json')
                                input_fp_list.append(cat + '/data/' + dataset_name + '/' + dataset_type + '.csv')
                                system_prompt_list.append(dataset_prompt)
                                batch_size_list.append(batch_size)

S="#!/bin/bash\n"
S+="# This script submits jobs sequentially, with each job depending on completion of the previous one\n\n"

# Add common slurm parameters
S+="# Common SLURM parameters\n"
S+="JOB_NAME=\"--job-name=infer_seq\"\n"
S+="NTASKS=\"--ntasks=1\"\n"
S+="TIME=\"--time=12:00:00\"\n"
S+="GRES=\"--gres=gpu:1\"\n\n"
# Generate the individual jobs with dependencies
S+="# Submit first job\n"
S+="cd ../\n"
S+="MODEL_ID=\"" + model_id_list[0] + "\"\n"
S+="OUTPUT_FP=\"" + output_fp_list[0] + "\"\n"
S+="INPUT_FP=\"" + input_fp_list[0] + "\"\n"
S+="SYSTEM_PROMPT=\"" + system_prompt_list[0] + "\"\n"
S+="BATCH_SIZE=\"" + str(batch_size_list[0]) + "\"\n\n"

S+="FIRST_JOB_ID=$(sbatch $MAIL_TYPE $MAIL_USER $JOB_NAME $NTASKS $TIME $PARTITION $GRES --parsable "
S+="--wrap=\"source ../myenv/bin/activate && python infer_lmentry.py --model_id \\\"$MODEL_ID\\\" --output_fp \\\"$OUTPUT_FP\\\" --input_fp \\\"$INPUT_FP\\\" --system_prompt \\\"$SYSTEM_PROMPT\\\" --batch_size $BATCH_SIZE && deactivate\")\n"
S+="echo \"Submitted job 1 with ID: $FIRST_JOB_ID\"\n\n"

S+="# Last submitted job ID\n"
S+="LAST_JOB_ID=$FIRST_JOB_ID\n\n"

S+="# Submit the rest of the jobs with dependencies\n"
for i in range(1, len(model_id_list)):
    S+=f"# Submit job {i+1}\n"
    S+=f"MODEL_ID=\"{model_id_list[i]}\"\n"
    S+=f"OUTPUT_FP=\"{output_fp_list[i]}\"\n"
    S+=f"INPUT_FP=\"{input_fp_list[i]}\"\n"
    S+=f"SYSTEM_PROMPT=\"{system_prompt_list[i]}\"\n"
    S+=f"BATCH_SIZE=\"{batch_size_list[i]}\"\n\n"
    
    S+=f"JOB_ID=$(sbatch $MAIL_TYPE $MAIL_USER $JOB_NAME $NTASKS $TIME $PARTITION $GRES --dependency=afterok:$LAST_JOB_ID --parsable "
    S+="--wrap=\"source ../myenv/bin/activate && python infer_lmentry.py --model_id \\\"$MODEL_ID\\\" --output_fp \\\"$OUTPUT_FP\\\" --input_fp \\\"$INPUT_FP\\\" --system_prompt \\\"$SYSTEM_PROMPT\\\" --batch_size $BATCH_SIZE && deactivate\")\n"
    S+=f"echo \"Submitted job {i+1} with ID: $JOB_ID\"\n"
    S+="LAST_JOB_ID=$JOB_ID\n\n"

S+="echo \"All jobs submitted sequentially. Final job ID: $LAST_JOB_ID\"\n"

dest_dir='./'
f= open(dest_dir+slurm_file,"w+")
f.write(S)
f.close()