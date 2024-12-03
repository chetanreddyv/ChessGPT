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
def parse_chess_games(file_path, limit=5000):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    count = 0
    for line in lines:
        if "###" in line:
            if count >= limit:
                break
            # Extract Elo ratings from the metadata
            parts = line.split("###")
            metadata = parts[0].strip()
            raw_moves = parts[1].strip()
            try:
                meta_parts = metadata.split()
                welo = int(meta_parts[3]) if meta_parts[3] != "None" else 0
                belo = int(meta_parts[4]) if meta_parts[4] != "None" else 0
            except (IndexError, ValueError):
                continue

            # Filter based on Elo ratings
            if welo < 2500 or belo < 2500:
                continue

            # Extract moves
            moves = re.findall(r"\.\s*([^\.]+)", raw_moves)
            if not moves:
                continue

            game_sequence = " ".join(moves).strip()
            if game_sequence:
                data.append(game_sequence)
                count += 1

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
        move_obj = board.parse_san(move)
        return True
    except ValueError:
        return False

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# Data Preparation
file_path = "/content/all_with_filtered_anotations_since1998 copy.txt"  # Update with your actual data file path
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
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
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
)

# Fine-Tuning
trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

def predict_next_move(model, tokenizer, board, move_history):
    device = next(model.parameters()).device

    # Create context with move history
    input_text = " ".join(move_history)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate multiple candidates and take first legal one
    outputs = model.generate(
        inputs.input_ids,
        max_length=inputs.input_ids.shape[1] + 5,  # Allow some tokens for the move
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    # Try each predicted move until finding a legal one
    for output in outputs:
        predicted_tokens = output[inputs.input_ids.shape[1]:]
        predicted_move = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip().split()[0]
        if is_legal_move(board, predicted_move):
            return predicted_move

    return "resign"

# Test the System
def play_game():
    board = chess.Board()
    move_history = []
    move_counter = 0
    max_moves = 100  # Prevent infinite loops
    
    # Load the fine-tuned model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    while not board.is_game_over() and move_counter < max_moves:
        print(board)
        print(f"\nMove {move_counter + 1}")

        if board.turn == chess.WHITE:
            move = predict_next_move(model, tokenizer, board, move_history)
            print(f"Predicted Move: {move}")

            if move != "resign":
                try:
                    board.push_san(move)
                    move_history.append(move)
                    move_counter += 1
                except ValueError:
                    print("Invalid move predicted by the model, resigning...")
                    break
            else:
                print("Model resigns.")
                break
        else:
            while True:
                try:
                    opponent_move = input("Opponent's move (or 'quit' to end): ")
                    if opponent_move.lower() == 'quit':
                        return
                    board.push_san(opponent_move)
                    move_history.append(opponent_move)
                    move_counter += 1
                    break
                except ValueError:
                    print("Invalid move, try again...")

    print("\nGame Over")
    print(f"Final position:\n{board}")
    print(f"Game history: {' '.join(move_history)}")

if __name__ == "__main__":
    play_game()