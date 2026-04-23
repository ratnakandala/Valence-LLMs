# Valence-LLMs

Code for the paper:
Ratna Kandala & Katie Hoemann, LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives,
NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle — Benchmarks, Emergent Abilities, and Scaling,
San Diego, California, USA, December 2025.

Paper: https://arxiv.org/pdf/2511.07641

\---

## Overview

This project evaluates how well three large language models (LLMs) estimate the emotional valence of Belgian-Dutch experience sampling narratives, compared to traditional sentiment tools (LIWC and Pattern). Valence is rated on a continuous scale from -50 (extremely unpleasant) to +50 (extremely pleasant).

\---

## Repository Structure

```
├── Valence\_estimation\_ChocoLlama.py     # Valence inference using ChocoLlama-8B-instruct
├── Valence\_estimation\_GEITje.py         # Valence inference using GEITje-7B-ultra
├── Valence\_estimation\_Reynardae.py      # Valence inference using Reynaerde-7B-Chat
├── valence\_analysis\_LLMs.R              # Correlations and visualisations (LLMs vs user valence)
├── valence\_correlations.R               # Pearson, polyserial, rmcorr, and mixed models (LIWC/Pattern vs LLMs)
└── README.md
```

\---

## Models

|File|Model|Loading|
|-|-|-|
|`Valence\_estimation\_ChocoLlama.py`|[ChocoLlama/Llama-3-ChocoLlama-8B-instruct](https://huggingface.co/ChocoLlama/Llama-3-ChocoLlama-8B-instruct)|Full precision, single GPU|
|`Valence\_estimation\_Reynardae.py`|[ReBatch/Reynaerde-7B-Chat](https://huggingface.co/ReBatch/Reynaerde-7B-Chat)|4-bit quantization (BitsAndBytes), batched|
|`Valence\_estimation\_GEITje.py`|[BramVanroy/GEITje-7B-ultra](https://huggingface.co/BramVanroy/GEITje-7B-ultra)|8-bit quantization, batched|

\---

## Python Scripts

Each script follows the same structure:

1. Load the model and tokenizer
2. Define prompt templates (English zero-shot, Dutch zero-shot, few-shot)
3. Run inference on the full dataset
4. Save results to CSV

**To run:**

```bash
python Valence\_estimation\_ChocoLlama.py
```

For Reynaerde and GEITje, you can also pass arguments from the command line:

```bash
python Valence\_estimation\_Reynardae.py --input\_csv path/to/data.csv --text\_column your\_text\_column
python Valence\_estimation\_GEITje.py --input\_csv path/to/data.csv --text\_column your\_text\_column
```

**Switching prompts:** Each script defines three prompt templates. To switch, pass the desired function in the `progress\_apply` (ChocoLlama) or `calculate\_batch\_scores` (Reynaerde, GEITje) call:

```python
# Replace zero\_shot\_prompt\_english with zero\_shot\_prompt\_dutch or few\_shot\_prompt as needed
```

**Requirements:**

```
torch
torchvision
transformers
bitsandbytes
pandas
numpy
scipy
scikit-learn
tqdm
python-dotenv
```

\---

## R Scripts

### `valence\_analysis\_LLMs.R`

* Loads valence estimates from all three LLMs alongside LIWC and Pattern outputs
* Computes Pearson correlations (user valence vs LIWC/Pattern) and polyserial correlations (user valence vs LLM estimates, LIWC/Pattern vs LLM estimates)
* Generates distribution plots, scatter plots with linear fits, and boxplots for each model
* Fits a linear mixed-effects model predicting user valence from Pattern polarity

### `valence\_correlations.R`

* Computes Pearson and polyserial correlations between user valence, LIWC (posemo, negemo), Pattern polarity, and LLM estimates
* Compares correlation coefficients using `paired.r` (psych) and `cocor`
* Computes repeated measures correlations using `rmcorr`
* Fits linear mixed-effects models (lme4) predicting user valence from LIWC and Pattern scores

**Required R packages:**

```r
tidyverse, psych, vroom, corrplot, readr, lme4, lmerTest,
CorrMixed, polycor, cocor, rmcorr, tidyr, Rfast, ggplot2
```

\---

## Input Data

Each script expects a CSV file with at minimum:

* A column of text responses (lemmatized Belgian-Dutch narratives)
* A `valence` column with participant self-reported valence ratings (-50 to +50)

Update the path and column name placeholders in each script before running.

\---

## Citation

```bibtex
@inproceedings{kandala2025llms,
  title     = {LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives},
  author    = {Kandala, Ratna and Hoemann, Katie},
  booktitle = {NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle},
  year      = {2025},
  address   = {San Diego, California, USA},
  url       = {https://arxiv.org/pdf/2511.07641}
}
```

