import chess
import chess.engine
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import random
import torch

# Initialize the chess board
board = chess.Board()

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to fine-tune the model on FEN data
def fine_tune_model(model, tokenizer, dataset_path):
    # Load the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128
    )
    
    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Fine-tune the model
    trainer.train()

# Fine-tune the model on your dataset (replace 'fen_dataset.txt' with your dataset file)
fine_tune_model(model, tokenizer, 'fen_dataset.txt')

def generate_move(board, model, tokenizer):
    # Convert the board to a string representation
    board_str = board.fen()
    
    # Encode the board state
    inputs = tokenizer.encode(board_str, return_tensors="pt")
    
    # Generate the next move
    outputs = model.generate(inputs, max_length=inputs.shape[1] + 10, num_return_sequences=1)
    
    # Decode the generated move
    move_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Attempt to extract a move from the generated text using regex
    import re
    match = re.search(r"([a-h][1-8][a-h][1-8])", move_str)
    if match:
        move = match.group(1)
        return move
    
    # If no valid move found in the text, return a random legal move
    move = random.choice(list(board.legal_moves)).uci()
    
    return move

# Play a game of chess
while not board.is_game_over():
    print(board)
    
    # Generate a move
    move = generate_move(board, model, tokenizer)
    
    # Make the move
    if chess.Move.from_uci(move) in board.legal_moves:
        board.push(chess.Move.from_uci(move))
    else:
        print(f"Illegal move generated: {move}")
        break

    # Print the board
    print(board)

    # Check if the game is over
    if board.is_game_over():
        break

    # Opponent's move (for simplicity, we use a random legal move)
    opponent_move = random.choice(list(board.legal_moves))
    board.push(opponent_move)

print("Game over")
print(board.result())