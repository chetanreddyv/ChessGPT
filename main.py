import chess
import chess.pgn
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator

# Dataset Preparation - Parse Chess Game Moves as Sequences
def parse_chess_games(file_path, limit=50000):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for count, line in enumerate(lines):
        if "###" in line:
            if count >= limit:
                break

            # Extract Elo ratings from the metadata
            metadata = line.split("###")[0].strip()
            try:
                welo, belo = metadata.split()[3:5]
                welo = int(welo) if welo != "None" else 0
                belo = int(belo) if belo != "None" else 0
            except (IndexError, ValueError):
                continue

            # Filter based on Elo ratings
            if welo < 2500 or belo < 2500:
                continue

            # Extract moves
            raw_moves = line.split("###")[1].strip()
            moves = re.findall(r"\.\s*([^\s]+)", raw_moves)
            if not moves:
                continue

            game_sequence = " ".join(moves).strip()
            if game_sequence:
                data.append(game_sequence)

    return pd.DataFrame(data, columns=["text"])

# Dataset Class for Language Modeling
class ChessDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# Function to Check Move Legality
def is_legal_move(board, move):
    try:
        board.push_san(move)
        board.pop()
        return True
    except ValueError:
        return False

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Include num_items_in_batch argument
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        labels = inputs["labels"]  # Shape: (batch_size, seq_len)

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Penalize illegal moves
        batch_size, seq_len = shift_labels.shape
        board = chess.Board()
        for i in range(batch_size):
            move_sequence = []
            for j in range(seq_len):
                token = shift_labels[i, j].item()
                if token == -100:  # Skip padding
                    continue
                move = tokenizer.decode([token]).strip()
                if not is_legal_move(board, move):
                    loss[i * seq_len + j] *= 2  # Increase penalty for illegal moves
                else:
                    board.push_san(move)  # Apply the move to the board

        return (loss.mean(), outputs) if return_outputs else loss.mean()

# Data Preparation
file_path = "/content/all_with_filtered_anotations_since1998 copy.txt"
raw_data = parse_chess_games(file_path)
train_data = raw_data.sample(frac=0.8, random_state=42)
val_data = raw_data.drop(train_data.index)

# Tokenizer and Model Setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA Configuration
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1
)
model = get_peft_model(base_model, lora_config)

# Dataset Objects
train_dataset = ChessDataset(train_data, tokenizer)
val_dataset = ChessDataset(val_data, tokenizer)

# Accelerator Setup
accelerator = Accelerator()

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
)

# Trainer Initialization
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-Tuning
trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Inference Pipeline
def predict_next_move(model, tokenizer, board, move_history, color):
    input_text = " ".join(move_history)
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=len(inputs[0]) + 1, pad_token_id=tokenizer.eos_token_id)
    predicted_move = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
    if is_legal_move(board, predicted_move):
        return predicted_move
    return "Illegal Move"

# Test the System
board = chess.Board()
move_history = []
color = "White"

while not board.is_game_over():
    print(board)
    if board.turn == chess.WHITE and color == "White" or board.turn == chess.BLACK and color == "Black":
        move = predict_next_move(model, tokenizer, board, move_history, color)
        print(f"Predicted Move: {move}")
        if move != "Illegal Move":
            board.push_san(move)
            move_history.append(move)
        else:
            print("Illegal move predicted, skipping...")
    else:
        opponent_move = input("Opponent's move: ")
        board.push_san(opponent_move)
        move_history.append(opponent_move)

print("Game Over")
