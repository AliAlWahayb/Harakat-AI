import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import pandas as pd
import json
import pickle
import time
import glob
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import copy
import logging
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

# Import from your existing module
from arabic_diacritics_data_processor import (
    ArabicDiacriticsDataProcessor, 
    ArabicDiacriticsDataset, 
    SimplifiedArabicDiacritizationModel,
    load_csv_data_from_directory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedTrainer")

class AdvancedEnsembleTrainer:
    """
    Advanced trainer that implements ensemble learning, mixed precision training,
    learning rate scheduling, and other advanced techniques for Arabic diacritization.
    """
    
    def __init__(
        self,
        model_paths,
        processor_paths=None,
        data_directory="data/",
        results_dir="results/advanced_ensemble",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model_paths = model_paths
        self.processor_paths = processor_paths
        self.data_directory = data_directory
        self.results_dir = results_dir
        self.device = device
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize models and processors
        self.models = []
        self.processors = []
        
        logger.info(f"Initializing {len(model_paths)} models for ensemble training")
        
        for i, model_path in enumerate(model_paths):
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get hyperparameters
            hyperparams = checkpoint.get('hyperparameters', {})
            max_sequence_length = hyperparams.get('max_sequence_length', 256)
            embedding_dim = hyperparams.get('embedding_dim', 256)
            hidden_dim = hyperparams.get('hidden_dim', 512)
            dropout_rate = hyperparams.get('dropout_rate', 0.3)
            
            # Load processor
            if processor_paths and i < len(processor_paths):
                with open(processor_paths[i], 'rb') as f:
                    processor = pickle.load(f)
            else:
                processor = ArabicDiacriticsDataProcessor(max_sequence_length=max_sequence_length)
            
            # Create model
            model = SimplifiedArabicDiacritizationModel(
                vocab_size=len(processor.char_to_idx),
                diacritic_size=len(processor.diacritic_to_idx),
                max_sequence_length=max_sequence_length,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            self.models.append(model)
            self.processors.append(processor)
            
            logger.info(f"Model {i+1} loaded: {model_path}")
            logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize ensemble model
        self.ensemble_model = None
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'character_accuracy': [],
            'word_accuracy': [],
            'diacritic_error_rate': []
        }
    
    def load_and_prepare_data(self, validation_split=0.1, test_split=0.1, augmentation_factor=0.2):
        """Load data from CSV files and prepare datasets"""
        logger.info(f"Loading data from {self.data_directory}")
        
        # Load diacritized texts
        diacritized_texts = load_csv_data_from_directory(self.data_directory)
        logger.info(f"Loaded {len(diacritized_texts)} text samples")
        
        # Use the first processor for data preparation
        processor = self.processors[0]
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        inputs, targets = processor.prepare_dataset(diacritized_texts)
        
        # Apply data augmentation
        logger.info(f"Applying data augmentation with factor {augmentation_factor}...")
        augmented_inputs, augmented_targets = processor.data_augmentation(
            inputs, targets, augmentation_factor=augmentation_factor
        )
        
        # Split dataset
        logger.info("Splitting dataset...")
        # First split: separate test set
        inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
            augmented_inputs, augmented_targets, test_size=test_split, random_state=42
        )
        
        # Second split: separate validation set from training set
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(
            inputs_train_val, targets_train_val, test_size=validation_split, random_state=42
        )
        
        logger.info(f"Train set: {len(inputs_train)} samples")
        logger.info(f"Validation set: {len(inputs_val)} samples")
        logger.info(f"Test set: {len(inputs_test)} samples")
        
        # Create PyTorch datasets
        train_dataset = ArabicDiacriticsDataset(inputs_train, targets_train, processor)
        val_dataset = ArabicDiacriticsDataset(inputs_val, targets_val, processor)
        test_dataset = ArabicDiacriticsDataset(inputs_test, targets_test, processor)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Store raw data for evaluation
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        
        return train_dataset, val_dataset, test_dataset
    
    def create_ensemble_model(self, ensemble_type="weighted_average"):
        """Create an ensemble model from the loaded models"""
        logger.info(f"Creating ensemble model with type: {ensemble_type}")
        
        if ensemble_type == "weighted_average":
            # Create a weighted average ensemble
            self.ensemble_model = WeightedAverageEnsemble(
                models=self.models,
                device=self.device
            )
        elif ensemble_type == "stacked":
            # Create a stacked ensemble
            self.ensemble_model = StackedEnsemble(
                models=self.models,
                diacritic_size=len(self.processors[0].diacritic_to_idx),
                device=self.device
            )
        elif ensemble_type == "boosted":
            # Create a boosted ensemble
            self.ensemble_model = BoostedEnsemble(
                models=self.models,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return self.ensemble_model
    
    def train_ensemble(
        self,
        batch_size=32,
        learning_rate=1e-4,
        epochs=20,
        patience=5,
        ensemble_type="weighted_average",
        use_mixed_precision=True,
        scheduler_type="one_cycle",
        weight_decay=1e-5,
        gradient_accumulation_steps=1
    ):
        """Train the ensemble model"""
        # Create ensemble model if not already created
        if self.ensemble_model is None:
            self.create_ensemble_model(ensemble_type)
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize optimizer
        if ensemble_type == "weighted_average":
            # Only optimize the weights
            optimizer = optim.Adam(
                self.ensemble_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            # Optimize all parameters
            optimizer = optim.AdamW(
                self.ensemble_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Initialize scheduler
        if scheduler_type == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,
                T_mult=2,
                eta_min=learning_rate / 100
            )
        else:
            scheduler = None
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler() if use_mixed_precision else None
        
        # Initialize best validation loss and patience counter
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        logger.info(f"Starting ensemble training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.ensemble_model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                # Move batch to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass with mixed precision
                if use_mixed_precision:
                    with autocast():
                        outputs = self.ensemble_model(inputs)
                        
                        # Reshape for loss calculation
                        batch_size, seq_len, num_classes = outputs.shape
                        outputs_flat = outputs.reshape(-1, num_classes)
                        targets_flat = targets.reshape(-1)
                        
                        # Calculate loss
                        loss = criterion(outputs_flat, targets_flat) / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        if scheduler is not None:
                            scheduler.step()
                else:
                    # Standard training without mixed precision
                    outputs = self.ensemble_model(inputs)
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, num_classes = outputs.shape
                    outputs_flat = outputs.reshape(-1, num_classes)
                    targets_flat = targets.reshape(-1)
                    
                    # Calculate loss
                    loss = criterion(outputs_flat, targets_flat) / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if scheduler is not None:
                            scheduler.step()
                
                train_loss += loss.item() * gradient_accumulation_steps
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            self.metrics_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.ensemble_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    # Forward pass
                    outputs = self.ensemble_model(inputs)
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, num_classes = outputs.shape
                    outputs_flat = outputs.reshape(-1, num_classes)
                    targets_flat = targets.reshape(-1)
                    
                    # Calculate loss
                    loss = criterion(outputs_flat, targets_flat)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, dim=2)
                    mask = (targets != 0)  # Ignore padding
                    correct += ((predicted == targets) & mask).sum().item()
                    total += mask.sum().item()
            
            # Calculate average validation loss and accuracy
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total if total > 0 else 0
            
            self.metrics_history['val_loss'].append(avg_val_loss)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Accuracy: {accuracy:.4f}")
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self.save_ensemble_model(f"{self.results_dir}/best_ensemble_model.pt")
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for evaluation
        self.load_ensemble_model(f"{self.results_dir}/best_ensemble_model.pt")
        
        # Evaluate on test set
        metrics = self.evaluate_ensemble()
        
        # Save training history
        self.save_training_history()
        
        return metrics
    
    def evaluate_ensemble(self, batch_size=32):
        """Evaluate the ensemble model on the test set"""
        logger.info("Evaluating ensemble model on test set")
        
        # Create test dataloader
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        # Set model to evaluation mode
        self.ensemble_model.eval()
        
        # Get model predictions
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                inputs = batch['input'].to(self.device)
                
                # Forward pass
                outputs = self.ensemble_model(inputs)
                
                # Get predictions
                _, predictions = torch.max(outputs, dim=2)
                all_predictions.extend(predictions.cpu().numpy())
        
        # Use the first processor for evaluation
        processor = self.processors[0]
        
        # Convert indices to diacritics
        true_diacritics = []
        pred_diacritics = []
        
        for i, (input_text, target) in enumerate(zip(self.inputs_test, self.targets_test)):
            # Truncate to actual length
            length = min(len(input_text), processor.max_sequence_length)
            
            # Convert target indices to diacritics
            true_diac = processor.decode_diacritics(
                processor.encode_target(target, pad=False)[:length]
            )
            
            # Convert predicted indices to diacritics
            pred_diac = processor.decode_diacritics(
                all_predictions[i][:length]
            )
            
            true_diacritics.append(true_diac)
            pred_diacritics.append(pred_diac)
        
        # Calculate metrics
        char_accuracy = self.character_level_accuracy(true_diacritics, pred_diacritics)
        word_accuracy = self.word_level_accuracy(self.inputs_test, true_diacritics, pred_diacritics)
        der = self.diacritic_error_rate(true_diacritics, pred_diacritics)
        
        # Store metrics
        self.metrics_history['character_accuracy'].append(char_accuracy)
        self.metrics_history['word_accuracy'].append(word_accuracy)
        self.metrics_history['diacritic_error_rate'].append(der)
        
        # Print metrics
        logger.info(f"Character-level accuracy: {char_accuracy:.4f}")
        logger.info(f"Word-level accuracy: {word_accuracy:.4f}")
        logger.info(f"Diacritic error rate: {der:.4f}")
        
        # Show example predictions
        self.show_example_predictions(self.inputs_test[:5], true_diacritics[:5], pred_diacritics[:5])
        
        # Save metrics
        metrics = {
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'diacritic_error_rate': der
        }
        
        with open(f"{self.results_dir}/ensemble_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def character_level_accuracy(self, true_diacritics, pred_diacritics):
        """Calculate character-level accuracy"""
        correct = 0
        total = 0
        
        for true_seq, pred_seq in zip(true_diacritics, pred_diacritics):
            for true_diac, pred_diac in zip(true_seq, pred_seq):
                if true_diac == pred_diac:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0
    
    def word_level_accuracy(self, input_texts, true_diacritics, pred_diacritics):
        """Calculate word-level accuracy"""
        correct_words = 0
        total_words = 0
        
        for text, true_seq, pred_seq in zip(input_texts, true_diacritics, pred_diacritics):
            # Split text into words
            words = text.split()
            
            # Track position in the character sequence
            pos = 0
            
            for word in words:
                word_length = len(word)
                
                # Extract diacritics for this word
                true_word_diacritics = true_seq[pos:pos+word_length]
                pred_word_diacritics = pred_seq[pos:pos+word_length]
                
                # Check if all diacritics match
                if true_word_diacritics == pred_word_diacritics:
                    correct_words += 1
                
                total_words += 1
                pos += word_length + 1  # +1 for space
        
        return correct_words / total_words if total_words > 0 else 0
    
    def diacritic_error_rate(self, true_diacritics, pred_diacritics):
        """Calculate diacritic error rate (DER)"""
        errors = 0
        total = 0
        
        for true_seq, pred_seq in zip(true_diacritics, pred_diacritics):
            for true_diac, pred_diac in zip(true_seq, pred_seq):
                if true_diac != pred_diac and true_diac != '':  # Only count errors on actual diacritics
                    errors += 1
                if true_diac != '':
                    total += 1
        
        return errors / total if total > 0 else 0
    
    def show_example_predictions(self, inputs, true_diacritics, pred_diacritics):
        """Show example predictions"""
        logger.info("\nExample Predictions:")
        logger.info("-" * 80)
        
        # Use the first processor for applying diacritics
        processor = self.processors[0]
        
        for i, (input_text, true_diac, pred_diac) in enumerate(zip(inputs, true_diacritics, pred_diacritics)):
            # Apply diacritics to input text
            true_text = processor.apply_diacritics(input_text, true_diac)
            pred_text = processor.apply_diacritics(input_text, pred_diac)
            
            logger.info(f"Example {i+1}:")
            logger.info(f"Input:      {input_text}")
            logger.info(f"True:       {true_text}")
            logger.info(f"Predicted:  {pred_text}")
            logger.info(f"Accuracy:   {sum(t == p for t, p in zip(true_diac, pred_diac)) / len(true_diac):.2f}")
            logger.info("-" * 80)
    
    def save_ensemble_model(self, path):
        """Save the ensemble model"""
        torch.save({
            'model_state_dict': self.ensemble_model.state_dict(),
            'ensemble_type': self.ensemble_model.__class__.__name__,
            'processors': self.processors
        }, path)
    
    def load_ensemble_model(self, path):
        """Load the ensemble model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        return self.ensemble_model
    
    def save_training_history(self):
        """Save training history and create plots"""
        # Save metrics history
        with open(f"{self.results_dir}/training_history.json", 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        # Create plots directory
        plots_dir = f"{self.results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plots_dir}/loss_history.png", dpi=300)
        plt.close()
        
        # Plot accuracy metrics
        if len(self.metrics_history['character_accuracy']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_history['character_accuracy'], label='Character Accuracy')
            plt.plot(self.metrics_history['word_accuracy'], label='Word Accuracy')
            plt.plot(self.metrics_history['diacritic_error_rate'], label='DER')
            plt.xlabel('Evaluation')
            plt.ylabel('Metric Value')
            plt.title('Accuracy Metrics')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_dir}/accuracy_metrics.png", dpi=300)
            plt.close()


class WeightedAverageEnsemble(nn.Module):
    """
    Ensemble model that computes a weighted average of the outputs of multiple models.
    """
    
    def __init__(self, models, device):
        super(WeightedAverageEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        
        # Initialize weights for each model
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x):
        # Get outputs from all models
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Apply softmax to weights
        weights = torch.softmax(self.weights, dim=0)
        
        # Compute weighted average
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output


class StackedEnsemble(nn.Module):
    """
    Stacked ensemble model that uses the outputs of multiple models as input to a meta-model.
    """
    
    def __init__(self, models, diacritic_size, device):
        super(StackedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        
        # Freeze base models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Meta-model (a simple MLP)
        self.meta_model = nn.Sequential(
            nn.Linear(len(models) * diacritic_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, diacritic_size)
        )
    
    def forward(self, x):
        # Get outputs from all models
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Concatenate outputs
        batch_size, seq_len, diacritic_size = outputs[0].shape
        stacked_outputs = torch.cat(outputs, dim=2)
        
        # Reshape for meta-model
        reshaped = stacked_outputs.view(batch_size * seq_len, -1)
        
        # Apply meta-model
        meta_output = self.meta_model(reshaped)
        
        # Reshape back
        ensemble_output = meta_output.view(batch_size, seq_len, -1)
        
        return ensemble_output


class BoostedEnsemble(nn.Module):
    """
    Boosted ensemble model that sequentially applies models, with each model
    focusing on the errors of the previous models.
    """
    
    def __init__(self, models, device):
        super(BoostedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        
        # Initialize weights for each model
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x):
        # Get outputs from all models
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Apply softmax to weights
        weights = torch.softmax(self.weights, dim=0)
        
        # Compute weighted sum
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output


def find_top_models(results_dir, top_n=3):
    """Find the top N models based on character accuracy"""
    # Find all result JSON files
    result_files = glob.glob(f"{results_dir}/**/results.json", recursive=True)
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    # Load results
    all_results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {str(e)}")
    
    # Sort by character accuracy
    sorted_results = sorted(
        all_results, 
        key=lambda x: x['metrics']['character_accuracy'], 
        reverse=True
    )
    
    # Get top N models
    top_models = sorted_results[:top_n]
    
    return top_models


def main():
    """Main function to run advanced ensemble training"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Ensemble Training for Arabic Diacritization")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing model results")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="results/advanced_ensemble", help="Output directory")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top models to use in ensemble")
    parser.add_argument("--ensemble_type", type=str, default="weighted_average", 
                        choices=["weighted_average", "stacked", "boosted"], 
                        help="Type of ensemble to create")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--scheduler", type=str, default="one_cycle", 
                        choices=["one_cycle", "cosine", "none"], 
                        help="Learning rate scheduler")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--augmentation_factor", type=float, default=0.2, help="Data augmentation factor")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Find top models
    logger.info(f"Finding top {args.top_n} models from {args.results_dir}")
    top_models = find_top_models(args.results_dir, top_n=args.top_n)
    
    # Print top models
    logger.info("Top models:")
    for i, model in enumerate(top_models):
        logger.info(f"{i+1}. {model['model_name']}: {model['metrics']['character_accuracy']:.4f}")
    
    # Get model and processor paths
    model_paths = []
    processor_paths = []
    
    for model in top_models:
        model_name = model['model_name']
        model_path = f"models/{model_name}/{model_name}.pt"
        processor_path = f"models/{model_name}/{model_name}_processor.pkl"
        
        if os.path.exists(model_path):
            model_paths.append(model_path)
            
            if os.path.exists(processor_path):
                processor_paths.append(processor_path)
            else:
                logger.warning(f"Processor not found: {processor_path}")
        else:
            logger.warning(f"Model not found: {model_path}")
    
    if not model_paths:
        raise ValueError("No valid model paths found")
    
    # Create ensemble trainer
    trainer = AdvancedEnsembleTrainer(
        model_paths=model_paths,
        processor_paths=processor_paths if processor_paths else None,
        data_directory=args.data_dir,
        results_dir=args.output_dir,
        device=device
    )
    
    # Load and prepare data
    trainer.load_and_prepare_data(augmentation_factor=args.augmentation_factor)
    
    # Train ensemble
    metrics = trainer.train_ensemble(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        ensemble_type=args.ensemble_type,
        use_mixed_precision=not args.no_mixed_precision,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps
    )
    
    # Print final metrics
    logger.info("\nFinal Metrics:")
    logger.info(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
    logger.info(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
    logger.info(f"Diacritic Error Rate: {metrics['diacritic_error_rate']:.4f}")
    
    # Save final results
    results = {
        'ensemble_type': args.ensemble_type,
        'top_models': [model['model_name'] for model in top_models],
        'metrics': metrics,
        'hyperparameters': vars(args),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{args.output_dir}/final_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {args.output_dir}/final_results.json")


if __name__ == "__main__":
    main()