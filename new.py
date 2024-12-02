import os
import time
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the dataset
final_df = pd.read_csv('final_chess_games.csv')
final_df = final_df[['Result', 'AN']]

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

# Prepare dataset
print("Loading dataset...")
datasets = load_dataset('text', data_files={'train': 'chess_moves.txt'})

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, max_length=128)

tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# Training
print("Starting training...")
trainer.train()
print("Training completed.")

# Save the final model
trainer.save_model('./chess_gpt_model')
tokenizer.save_pretrained('./chess_gpt_model')

# Generate moves
print("Generating moves...")
model.eval()

def generate_moves(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    sample_outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
start_sequence = '1. e4 e5 2. Nf3 Nc6'
generated_moves = generate_moves(model, tokenizer, start_sequence)
print("Generated moves:", generated_moves)