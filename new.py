import os
import time
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load the dataset
final_df = pd.read_csv('final_chess_games.csv')
final_df = final_df[['Result', 'AN']].head(5000)  # Select first 5000 rows

# Prepare the text data
print("Preprocessing data...")
text_data = final_df['AN'].apply(lambda x: x.strip()).tolist()
with open('chess_moves.txt', 'w') as f:
    for line in text_data:
        f.write(line + '\n')
print("Data preprocessing completed.")

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Apply LoRA for parameter-efficient fine-tuning using PEFT
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,                # Rank of the low-rank decomposition
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.1,   # Dropout rate
    target_modules=["attn.c_proj"],  # Target layers for adaptation
)

model = get_peft_model(model, lora_config)

# Freeze all model parameters except LoRA parameters
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# Prepare dataset
print("Loading dataset...")
datasets = load_dataset('text', data_files={'train': 'chess_moves.txt'})

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, max_length=128, padding='max_length')

tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    eval_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision for memory efficiency
    optim='adamw_torch',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./lora_finetuned_gpt')