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

# Step 4: Split Data
train_data = raw_data[:400]
test_data = raw_data[400:]

# Step 5: Convert to HuggingFace Datasets
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Step 6: Load mBART Tokenizer & Model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set tokenizer to source and target languages
tokenizer.src_lang = "en_XX"
target_lang = "hi_IN"

# Step 7: Preprocessing Function
def preprocess_function(examples):
    inputs = tokenizer(examples["english"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples["hindi"], truncation=True, padding="max_length", max_length=128)

    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

import os
os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging
#Step 8
training_args = TrainingArguments(
    output_dir="/content/mbart-finetuned-hi",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    logging_dir="/content/logs"
)

# Step 9: Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# Step 10: Train
trainer.train()

# Step 11: Save Model
model.save_pretrained("/content/mbart-finetuned-hi")
tokenizer.save_pretrained("/content/mbart-finetuned-hi")
