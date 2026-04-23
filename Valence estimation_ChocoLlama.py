#Environment & version checks
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"}")

import torchvision
print(f"torchvision version: {torchvision.__version__}")

import transformers
print(f"transformers version: {transformers.__version__}")

#Standard library and third-party imports
import os
import re
import csv
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from transformers import AutoModelForCausalLM, AutoTokenizer 


#Load the ChocoLlama-8B-instruct tokenizer and model

start_time = time.time() #Start recording the time

tokenizer = AutoTokenizer.from_pretrained('ChocoLlama/Llama-3-ChocoLlama-8B-instruct')
model = AutoModelForCausalLM.from_pretrained('ChocoLlama/Llama-3-ChocoLlama-8B-instruct', low_cpu_mem_usage=True, device_map='auto')

end_time = time.time() #Stop recording the time
total_loading_time = end_time - start_time
print(f"Time taken to load the model: {total_loading_time:.2f} seconds")


#Load input data
DATA_PATH = r'path to the csv file'
TEXT_COLUMN_NAME = 'column name in the csv file'

df = pd.read_csv(DATA_PATH)
print(f"Shape of the input data: {df.shape}")
data = df[TEXT_COLUMN_NAME]


#------------------------------------------------------------------------------------------------------------
#PROMPT TEMPLATES
#------------------------------------------------------------------------------------------------------------
# Either one of these prompts can be used in the `calculate_valence_score` function by passing it as the `prompt` argument.

# ENGLISH prompt (zero-shot setting) — the model relies on its pre-existing understanding of valence
def zero_shot_prompt_english(text):
    return f"""
You are a native Belgian-Dutch speaker with expertise in assessing the emotional tone of daily narratives. \\
You will be provided with a text, and your task is to evaluate its overall valence on a scale from 1 to 7: 
 
1 means "extremely unpleasant" 
 
7 means "extremely pleasant" 
 
Please return only a single numerical rating, enclosed in square brackets, with no additional text, explanations, or formatting. 
 
Text: "{text}" 
 
Expected output: [Score]
"""

# DUTCH prompt (zero-shot setting) — the model relies on its pre-existing understanding of valence
def zero_shot_prompt_dutch(text):
    return f"""
    “Je bent een Nederlandse taalexpert die de valentie van Belgisch Nederlandse teksten analyseert.
    Deelnemers reageerden op: ‘Wat speelt er nu of sinds de vorige beep en hoe voel je je daarover?’
    Lees zorgvuldig het antwoord van de deelnemer: ‘{text}’.
    Het is jouw taak om het sentiment te beoordelen van 1 (zeer slecht) tot 7 (zeer goed). Geef alleen een
    cijfer tussen haakjes (bijv. [X]), zonder extra tekst, uitleg of opmaak. Niet uitleggen.
    Uitvoerformaat: [getal]. Vervang ‘getal’ door de gehele score (1–7).
    """

# FEW-SHOT prompt — includes example inputs and outputs to guide the model
def few_shot_prompt(text):
    return f"""
    "You are a Dutch language expert analyzing the valence of Belgian Dutch texts. Participants
    responded to the question:
    'What is going on now or since the last prompt, and how do you feel about it?'
    They also rated their own emotional valence on a continuous scale from -50 (very negative) to
    +50 (very positive). Your task is to read each response: {text} and rate its sentiment from 1 (very
    negative) to 7 (very positive). Return ONLY a single numerical rating enclosed in square
    brackets, e.g. [X]. Provide no explanation or additional text.
    Output Format: [number].
    Below are a few examples of an input and the participant's valence rating to guide your rating:
    Input: (Text1)
    Output: [10]
    Input: (Text2)
    Output: [-10]
    Input: (Text3)
    Output: [30]
    Input: (Text4)
    Output: [45]
    Input: (Text5)
    Output: [40]"
    """


#------------------------------------------------------------------------------------------------------------
#CORE INFERENCE FUNCTION (REUSABLE) - USED FOR ENGLISH ZERO-SHOT PROMPT
#------------------------------------------------------------------------------------------------------------

# The English and Dutch versions share the same logic — only the prompt differs.
# Pass the desired prompt string as the `prompt` argument.

def calculate_valence_score(
        text,
        prompt,
        tokenizer,
        model):
    
       
    #tokenize the prompt
    #input_ids is a tensor containing the tokenized input prompt
    #The tokenizer converts the text prompt into a format that the model can understand, 
    #The return_tensors="pt" argument specifies that the output should be in the form of PyTorch tensors. 
    #The .input_ids attribute extracts the token IDs from the tokenized output, and 
    #.to(model.device) moves the tensor to the same device (CPU or GPU) that the model is using for inference.
    
    input_ids = tokenizer(prompt(text), return_tensors = "pt").input_ids.to(model.device)

    #Generate the model's output - a short response (score only)
    outputs = model.generate(
        input_ids,
        max_new_tokens = 6, #restrict output length
        do_sample = False, #Disable sampling for deterministic output
        temperature = 0.0, #Remove randomness completely (increased values -> increased randomness)
        pad_token_id = tokenizer.eos_token_id #Prevent infinite generation
        )

    #Record the length of the prompt in tokens so we can isolate only the new ouptut below
    prompt_length = input_ids.shape[1] #Number of tokens in the input prompt
    decoded_output =tokenizer.decode(outputs[0], skip_special_tokens = True) #Decode the response converting token ids to text, skipping special tokens

    #Generate only the newly generated tokens
    new_tokens = outputs[0][prompt_length:]
    decoded_new_text = tokenizer.decode(new_tokens, skip_special_tokens = True)

    #Extract the score (first number within []) from the model's output
    try:
        match = re.search(r'\[?([+-]?\d+)\]?', decoded_new_text)
        if match is not None:
            valence_score = int(match.group(1)) #Extract the first capturing group (the number)
            valence_score = max(-50, min(50, valence_score)) #Clamp to valid range
        else:
            valence_score = None # No score found in the output
    except (AttributeError, ValueError):
        valence_score = None #Handle unexpected conversion errors gracefully

    print(f"Text: {text} | Valence score: {valence_score}") 

    return valence_score


#------------------------------------------------------------------------------------------------------------
#RUN INFERENCE ON THE ENTIRE DATASET
#------------------------------------------------------------------------------------------------------------
tqdm.pandas()
df['ChocoLlama_valence'] = data.progress_apply(
    lambda t: calculate_valence_score(t, prompt, tokenizer, model)
    )


#------------------------------------------------------------------------------------------------------------
#CALCULATE THE PEARSON CORRELATION VALUE, P-VALUE AND THE MEAN ABSOLUTE ERROR (MAE) BETWEEN THE MODEL'S PREDICTIONS AND THE HUMAN RATINGS
#------------------------------------------------------------------------------------------------------------
p_corr, p_val = pearsonr(df['ChocoLlama_valence'], df['valence']) #'valence' is the column in df that contains the human ratings
mae = mean_absolute_error(df['ChocoLlama_valence'], df['valence'])

print(f"Pearson Correlation: {p_corr}")
print(f"p value: {p_val}")
print(f"Mean Absolute Error: {mae}")


#------------------------------------------------------------------------------------------------------------
#VISUALISATION
#------------------------------------------------------------------------------------------------------------
plt.figure(figsize = (6,8))
plt.scatter(df['ChocoLlama_valence'], df['valence'])
plt.xlabel('ChocoLlama Valence')
plt.ylabel('Human Valence')
plt.title('Scatter Plot of LLM Valence vs Valence')
plt.grid(True)
plt.show()