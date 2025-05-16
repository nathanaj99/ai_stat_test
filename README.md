# "What Does Your Benchmark Really Measure? Towards Robust Inference of AI Capabilities"

## Python Packages
`requirements.txt` outlines all the python packages required.

## Scripts

The following describes the structure of our scripts

```
scripts
├── 1-preprocessing
│   ├── process_{dataset}.py
├── 2-model_inference
│   ├── bash_generator.py
│   ├── gpt_4.1.py
│   ├── small_models.py
├── 3-process_answers
├── 4-param_inference
│   ├── infer_accuracy_ability.py
│   ├── infer_irt_params.py
├── 5-other_analysis
```
The number represents roughly chronological ordering. Within each numbering, however, files do not have dependencies with the exception of (4), see below. Note that we provide all intermediate files and results from our data, so running (1)-(3) as well as `infer_irt_params.py` are not strictly necessary. Scripts in (4) and (5) produce figures which automatically save in `figs/`.

- `1-preprocessing`: processes the raw data
- `2-model_inference`: passes the questions into generative models to test their capabilities
- `3-process_answer`: takes the raw outputs from `model_inference` and evaluates correctness
- `4-param_inference`: run `infer_irt_params.py` (which infers item parameters) BEFORE `infer_accuracy_ability.py`, which infers the actual capability parameters of interest using Algorithm 1 and 2 from the paper.
- `5-other_analysis`: generates Figures 1, 2, 4, 5, and 6

## Data
Data is stored in the `files/` directory, which contain three subcategories: `bbh/`, `GPQA/`, and `lmentry/` -- the three benchmark data we use in our experiments. Within each subdirectory, `raw_data/` indicates data that has been pulled from other sources that created the dataset, while `data/` generally contains pre-processed files up to the point of model inference. `results/` shows both raw and processed results, categorized by each subtask and each language model's results. We now attribute the sources of `raw_data/`:
- `bbh/raw_data/original`: [Suzgun et al., 2022](https://arxiv.org/abs/2210.09261), access [data here](https://huggingface.co/datasets/lukaemon/bbh)
- `bbh/raw_data/perturbations`: [Mizrahi et al., 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00681/123885), access [data here](https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation)
- `GPQA/data/gpqa_main.csv`: [Rein et al., 2023](https://arxiv.org/abs/2311.12022), access [data here](https://github.com/idavidrein/gpqa/)
- `lmentry/raw_data/original`: [Efrat et al., 2023](https://aclanthology.org/2023.findings-acl.666.pdf), access [data here](https://github.com/aviaefrat/lmentry)
- `lmentry/raw_data/perturbations`: [Mizrahi et al., 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00681/123885), access [data here](https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation)