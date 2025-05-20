import os
import time
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pytorch_arabic_diacritization import (
    ArabicDiacriticsDataProcessor,
    ArabicDiacriticsDataset,
    SimplifiedArabicDiacritizationModel,
    DiacritizationEvaluator,
    load_csv_data_from_directory
)

# Constants
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 128
CHECKPOINT_PATH = './checkpoints/ADNN_512_128_256_512_0.3_0.001_500/ADNN_512_128_256_512_0.3_0.001_500.pt'  # Update to your checkpoint path
PROCESSOR_PATH = './models/ADNN_512_128_256_512_0.3_0.001_500/ADNN_512_128_256_512_0.3_0.001_500_processor.pkl'  # Update to your processor path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
def load_test_data(directory_path="data/"):
    """Load the test dataset for evaluation."""
    print("Loading test data from CSV files...")
    diacritized_texts = load_csv_data_from_directory(directory_path)
    return diacritized_texts

# Evaluate the model
def evaluate_model(model, processor, test_loader, test_inputs, test_targets):
    evaluator = DiacritizationEvaluator(model, processor)

    # Measure inference time
    inference_start_time = time.time()
    metrics = evaluator.evaluate(test_loader, test_inputs, test_targets)
    inference_time = time.time() - inference_start_time
    
    return metrics, inference_time, evaluator

# Main testing function
def test_model():
    # Load test data
    diacritized_texts = load_test_data()

    # Initialize data processor
    print("Initializing data processor...")
    data_processor = ArabicDiacriticsDataProcessor(max_sequence_length=MAX_SEQUENCE_LENGTH)

    # Prepare the dataset
    print("Preparing dataset...")
    inputs, targets = data_processor.prepare_dataset(diacritized_texts)
    
    # Split data (same as in training)
    inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
        inputs, targets, test_size=0.15, random_state=42
    )
    val_adjusted_size = 0.15 / (1 - 0.15)  # Adjust validation size
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs_train_val, targets_train_val, test_size=val_adjusted_size, random_state=42
    )

    print(f"Train set: {len(inputs_train)} samples")
    print(f"Validation set: {len(inputs_val)} samples")
    print(f"Test set: {len(inputs_test)} samples")

    # Create PyTorch datasets and dataloaders for testing
    test_dataset = ArabicDiacriticsDataset(inputs_test, targets_test, data_processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Load the trained model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SimplifiedArabicDiacritizationModel(
        vocab_size=len(data_processor.char_to_idx),
        diacritic_size=len(data_processor.diacritic_to_idx),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=256,
        hidden_dim=512,
        dropout_rate=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate the model
    metrics, inference_time, evaluator = evaluate_model(model, data_processor, test_loader, inputs_test, targets_test)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
    print(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
    print(f"Diacritic Error Rate: {metrics['diacritic_error_rate']:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Show example predictions with device-aware batching
    # We generate predictions for first 5 examples manually for display
    example_inputs = inputs_test[:50]
    example_targets = targets_test[:50]

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for text in example_inputs:
            encoded_input = data_processor.encode_input(text)
            input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)
            outputs = model(input_tensor)
            _, pred_indices = torch.max(outputs, dim=2)
            pred_diacritics = data_processor.decode_diacritics(pred_indices[0].cpu().numpy()[:len(text)])
            all_predictions.append(pred_diacritics)

    evaluator.show_example_predictions(example_inputs, example_targets, all_predictions)

if __name__ == "__main__":
    test_model()
