# Step 1: Install Required Libraries
!pip install transformers datasets sentencepiece
!pip install datasets

# Step 2: Import Modules
import json
import os
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, TrainingArguments, Trainer

# Step 3: Load & Prepare Data
with open("/content/english_hindi_slang.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)["data"]

# Add prompt-style context to input sentences
for item in raw_data:
    item["english"] = f'Translate the {item["context"]} meaning of: {item["english"]}'
