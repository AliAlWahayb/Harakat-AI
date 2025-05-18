import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import unicodedata
from tqdm import tqdm
import glob
import pickle
import time
import json
from datetime import datetime
import itertools
import math

# Constants
MAX_SEQUENCE_LENGTH = 256  # User can adjust this
BATCH_SIZE = 32 # Adjusted for potentially larger model
EMBEDDING_DIM = 256
HIDDEN_DIM = 512 # Used for FFN in Transformer
NUM_ENCODER_LAYERS = 4 # Transformer specific
NUM_ATTENTION_HEADS = 8 # Transformer specific
DROPOUT_RATE = 0.1 # Often lower in Transformers
LEARNING_RATE = 5e-5 # Common starting point for Transformers
EPOCHS = 50 # User can adjust, but with early stopping
PATIENCE = 7 # Increased patience

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ArabicDiacriticsDataProcessor:
    def __init__(self, max_sequence_length=MAX_SEQUENCE_LENGTH):
        self.max_sequence_length = max_sequence_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.diacritic_to_idx = {}
        self.idx_to_diacritic = {}
        
        # Define Arabic diacritics (ensure comprehensive list)
        self.diacritics = [
            '',        # No diacritic (PAD or end of effective diacritics)
            '\u064B',  # Fathatan
            '\u064C',  # Dammatan
            '\u064D',  # Kasratan
            '\u064E',  # Fatha
            '\u064F',  # Damma
            '\u0650',  # Kasra
            '\u0651',  # Shadda
            '\u0652',  # Sukun
            # Combined Shadda + Vowel (Treating as single units for simplification, can be expanded)
            '\u0651\u064E', # Shadda + Fatha
            '\u0651\u064F', # Shadda + Damma
            '\u0651\u0650', # Shadda + Kasra
            '\u0651\u064B', # Shadda + Fathatan
            '\u0651\u064C', # Shadda + Dammatan
            '\u0651\u064D', # Shadda + Kasratan
            '\u0651\u0652', # Shadda + Sukun (less common but possible)
            # Other less frequent but possible diacritics
            '\u0653',  # Maddah
            '\u0654',  # Hamza above
            '\u0655',  # Hamza below
            '\u0670',  # Superscript Alef
        ]
        
        # Initialize diacritic mappings
        for i, diac in enumerate(self.diacritics):
            self.diacritic_to_idx[diac] = i
            self.idx_to_diacritic[i] = diac
        
        # Ensure PAD diacritic is at index 0 if '' is used for padding
        self.pad_diacritic_idx = self.diacritic_to_idx.get('', 0)


    def strip_diacritics(self, text):
        """Remove diacritics from Arabic text while preserving characters"""
        # More robust stripping
        normalized_text = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized_text if not unicodedata.combining(c) and c not in self.diacritics)


    def extract_char_diacritic_pairs(self, text):
        chars = []
        diacs = []
        current_char = ''
        current_diacritics = []

        for char_code in text:
            if char_code in self.diacritics or unicodedata.combining(char_code): # Check if it's a diacritic
                # Handle Shadda combinations. If shadda is followed by another vowel, combine them.
                if char_code == '\u0651' and current_diacritics: # If shadda and previous char already has diacritic
                    pass # avoid double shadda or shadda on shadda
                elif char_code == '\u0651':
                    current_diacritics.append(char_code)
                elif current_diacritics and current_diacritics[-1] == '\u0651' and char_code in ['\u064E', '\u064F', '\u0650', '\u064B', '\u064C', '\u064D', '\u0652']:
                    # Combine Shadda with the vowel
                    combined_diac = current_diacritics.pop() + char_code
                    current_diacritics.append(combined_diac)
                else:
                    current_diacritics.append(char_code)
            else: # It's a base character
                if current_char: # If there was a previous character, store it with its diacritics
                    chars.append(current_char)
                    diacs.append("".join(current_diacritics) if current_diacritics else "")
                current_char = char_code
                current_diacritics = []
        
        # Add the last character and its diacritics
        if current_char:
            chars.append(current_char)
            diacs.append("".join(current_diacritics) if current_diacritics else "")
            
        return "".join(chars), diacs


    def create_character_mappings(self, texts):
        """Create character to index mappings from a list of texts"""
        unique_chars = set()
        for text in texts:
            for char_code in self.strip_diacritics(text): # Use stripped text for char vocabulary
                 unique_chars.add(char_code)
        
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(list(unique_chars)))} # 0 for PAD
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(self.char_to_idx) # Add UNK token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def prepare_dataset(self, diacritized_texts):
        inputs = []
        targets = []
        
        processed_texts = 0
        skipped_texts = 0

        for text in tqdm(diacritized_texts, desc="Preparing dataset"):
            if not isinstance(text, str): # Skip non-string entries
                skipped_texts +=1
                continue

            # Normalize text (e.g., Alef forms)
            text = unicodedata.normalize('NFC', text) # Use NFC for consistency
            
            # Further cleaning: remove non-Arabic characters except essential punctuation if needed
            # For simplicity, we'll keep it as is, but this can be a place for more aggressive cleaning.
            text = re.sub(r'[^\u0600-\u06FF\s\u0030-\u0039]', '', text) # Keep Arabic, spaces, numbers

            input_text, diacritics_sequence = self.extract_char_diacritic_pairs(text)
            
            if not input_text: # Skip if text becomes empty after processing
                skipped_texts +=1
                continue

            if len(input_text) != len(diacritics_sequence):
                # This check is crucial. If lengths mismatch, skip or try to fix.
                # print(f"Warning: Mismatch after extract_char_diacritic_pairs for text: '{text}'. Input len: {len(input_text)}, Diac len: {len(diacritics_sequence)}")
                # print(f"Input: {input_text}, Diacs: {diacritics_sequence}")
                skipped_texts +=1
                continue

            inputs.append(input_text)
            targets.append(diacritics_sequence)
            processed_texts +=1
        
        print(f"Processed {processed_texts} texts, Skipped {skipped_texts} texts.")
        if not inputs:
             raise ValueError("No valid data could be prepared. Check input data and processing logic.")

        self.create_character_mappings(inputs) # Create mappings from the processed inputs
        
        return inputs, targets
    
    def encode_input(self, text, pad=True):
        encoded = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        if len(encoded) > self.max_sequence_length:
            encoded = encoded[:self.max_sequence_length]
        
        if pad:
            padding_length = self.max_sequence_length - len(encoded)
            encoded.extend([self.char_to_idx['<PAD>']] * padding_length)
        return encoded
    
    def encode_target(self, diacritics, pad=True):
        # Ensure all diacritics are in the map, map unknowns to 'no diacritic'
        encoded = [self.diacritic_to_idx.get(d, self.diacritic_to_idx.get('', 0)) for d in diacritics]
        if len(encoded) > self.max_sequence_length:
            encoded = encoded[:self.max_sequence_length]

        if pad:
            padding_length = self.max_sequence_length - len(encoded)
            encoded.extend([self.pad_diacritic_idx] * padding_length) # Pad with 'no diacritic' index
        return encoded

    def decode_diacritics(self, indices):
        return [self.idx_to_diacritic.get(idx, '') for idx in indices] # Default to empty string if index is unknown

    def apply_diacritics(self, text, diacritics):
        result = []
        min_len = min(len(text), len(diacritics))
        for i in range(min_len):
            result.append(text[i])
            if diacritics[i] is not None and diacritics[i] != self.idx_to_diacritic.get(self.pad_diacritic_idx, ''): # Don't add pad diacritic
                 result.append(diacritics[i])
        
        # Append remaining text if any (should not happen if lengths are managed well)
        if len(text) > min_len:
            result.append(text[min_len:])
        return "".join(result)

    def data_augmentation(self, inputs, targets, augmentation_factor=0.1): # Reduced default factor
        augmented_inputs = []
        augmented_targets = []
        
        for input_text, target_diacritics in zip(inputs, targets):
            augmented_inputs.append(input_text)
            augmented_targets.append(target_diacritics)
            
            if random.random() < augmentation_factor:
                # 1. Random Diacritic Removal (partial diacritization)
                if len(target_diacritics) > 0:
                    new_target_diacritics = list(target_diacritics)
                    num_to_remove = random.randint(1, max(1, len(new_target_diacritics) // 5)) # Remove up to 20%
                    for _ in range(num_to_remove):
                        if any(d != '' for d in new_target_diacritics): # only remove if there's something to remove
                            idx_to_remove = random.choice([i for i, d in enumerate(new_target_diacritics) if d != ''])
                            new_target_diacritics[idx_to_remove] = ''
                    augmented_inputs.append(input_text)
                    augmented_targets.append(new_target_diacritics)

                # 2. Syntactic Noise (e.g., slight changes to common prefixes/suffixes if possible without changing meaning significantly)
                # This is complex and requires morphological awareness. For now, a simpler character swap:
                if len(input_text) > 3 and random.random() < 0.3: # Lower probability for this
                    char_map = {'ا': 'أ', 'أ': 'ا', 'إ': 'ا', 'ه': 'ة', 'ة': 'ه', 'ي': 'ى', 'ى': 'ي'}
                    pos = random.randint(0, len(input_text) - 1)
                    if input_text[pos] in char_map:
                        chars = list(input_text)
                        chars[pos] = char_map[input_text[pos]]
                        augmented_inputs.append(''.join(chars))
                        augmented_targets.append(target_diacritics) # Assume diacritics don't change with these minor swaps
        
        return augmented_inputs, augmented_targets


class ArabicDiacriticsDataset(Dataset):
    def __init__(self, inputs, targets, processor):
        self.inputs = inputs
        self.targets = targets
        self.processor = processor
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_diacritics = self.targets[idx]
        
        encoded_input = self.processor.encode_input(input_text)
        encoded_target = self.processor.encode_target(target_diacritics)
        
        # Actual length of the sequence before padding
        # This is important for loss calculation if using pack_padded_sequence or for evaluation
        actual_length = min(len(input_text), self.processor.max_sequence_length)

        return {
            'input': torch.tensor(encoded_input, dtype=torch.long),
            'target': torch.tensor(encoded_target, dtype=torch.long),
            'length': actual_length 
        }

# --- Transformer Model Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # Not a parameter, but should be part of state_dict

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # self.pe shape: [1, max_len, embedding_dim]
        # We need to select the positional encodings for the current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AdvancedArabicDiacritizationModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        diacritic_size, 
        d_model=EMBEDDING_DIM, # Renamed embedding_dim to d_model for Transformer convention
        nhead=NUM_ATTENTION_HEADS, 
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=HIDDEN_DIM, # For FFN in Transformer
        dropout=DROPOUT_RATE,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    ):
        super(AdvancedArabicDiacritizationModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_sequence_length)
        
        # Standard PyTorch TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Important: input format is [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Optional: CNN layers for local feature extraction before Transformer
        # This can sometimes help, especially if not using pre-trained embeddings.
        # For this version, we'll make it optional or try without first for simplicity.
        # self.use_cnn_stem = True # Or False
        # if self.use_cnn_stem:
        #     self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        #     self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        #     self.relu = nn.ReLU()
        #     self.layer_norm_cnn = nn.LayerNorm(d_model) # Normalize after CNN

        self.fc_out = nn.Linear(d_model, diacritic_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding_mask=None):
        # src shape: [batch_size, seq_len]
        # src_padding_mask shape: [batch_size, seq_len] where True means pad
        
        src = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding
        src = self.pos_encoder(src) # Add positional encoding
        
        # if self.use_cnn_stem:
        #     # CNN expects [batch, channels, seq_len]
        #     src_permuted = src.permute(0, 2, 1) 
        #     conv_out = self.relu(self.conv1(src_permuted) + self.conv2(src_permuted)) # Example: simple sum
        #     src = conv_out.permute(0, 2, 1) # Back to [batch, seq_len, d_model]
        #     src = self.layer_norm_cnn(src)

        # TransformerEncoder expects src_key_padding_mask
        # where True values are positions that should be ignored by attention.
        if src_padding_mask is None:
             src_padding_mask = (src_input == 0) # Assuming 0 is PAD index for input characters


        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
        # output shape: [batch_size, seq_len, d_model]
        
        logits = self.fc_out(output)
        # logits shape: [batch_size, seq_len, diacritic_size]
        return logits


class DiacritizationTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader=None, 
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5, # Added weight decay
        device=device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_loader.dataset.processor.pad_diacritic_idx) # Use processor's pad_diacritic_idx
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay) # AdamW is good for transformers
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=patience//2, min_lr=1e-7, verbose=True # More aggressive LR reduction
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Create source padding mask for the Transformer
            # True for padded positions, False for non-padded
            src_padding_mask = (inputs == self.train_loader.dataset.processor.char_to_idx['<PAD>']).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, src_padding_mask=src_padding_mask) # Pass the mask
            
            batch_size, seq_len, num_classes = outputs.shape
            loss = self.criterion(outputs.reshape(-1, num_classes), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total_elements = 0 # Total non-padded elements
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                src_padding_mask = (inputs == self.val_loader.dataset.processor.char_to_idx['<PAD>']).to(self.device)

                outputs = self.model(inputs, src_padding_mask=src_padding_mask)
                
                batch_size, seq_len, num_classes = outputs.shape
                loss = self.criterion(outputs.reshape(-1, num_classes), targets.reshape(-1))
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=2) # Get predictions along the class dimension
                
                # Create mask for non-padding target elements
                # Use processor's pad_diacritic_idx
                mask = (targets != self.val_loader.dataset.processor.pad_diacritic_idx).to(self.device)
                
                correct += ((predicted == targets) & mask).sum().item()
                total_elements += mask.sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_elements if total_elements > 0 else 0
        
        self.scheduler.step(val_loss) # Step scheduler based on validation loss
        
        return val_loss, accuracy
    
    def train(self, epochs=EPOCHS, checkpoint_path='checkpoints/pytorch_diacritization.pt', patience_epochs=PATIENCE):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        best_val_loss = float('inf')
        current_patience = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        print(f"Starting training for {epochs} epochs with patience {patience_epochs}.")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            log_message = f"Train Loss: {train_loss:.4f}"

            if self.val_loader is not None:
                val_loss, accuracy = self.validate()
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(accuracy)
                log_message += f", Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}"
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'accuracy': accuracy,
                        'char_to_idx': self.train_loader.dataset.processor.char_to_idx,
                        'idx_to_char': self.train_loader.dataset.processor.idx_to_char,
                        'diacritic_to_idx': self.train_loader.dataset.processor.diacritic_to_idx,
                        'idx_to_diacritic': self.train_loader.dataset.processor.idx_to_diacritic,
                        'max_sequence_length': self.train_loader.dataset.processor.max_sequence_length
                    }, checkpoint_path)
                    log_message += f" - Model Saved to {checkpoint_path}"
                else:
                    current_patience += 1
                    log_message += f" - Patience: {current_patience}/{patience_epochs}"
                    if current_patience >= patience_epochs:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break
            else: # No validation loader
                torch.save(self.model.state_dict(), checkpoint_path) # Save periodically or based on train loss
                log_message += f" - Model Saved (no validation) to {checkpoint_path}"

            epoch_duration = time.time() - epoch_start_time
            print(f"{log_message} (Epoch time: {epoch_duration:.2f}s)")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")


        if self.val_loader is not None and os.path.exists(checkpoint_path):
            print(f"Loading best model from {checkpoint_path} with val_loss: {best_val_loss:.4f}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model, history


class DiacritizationEvaluator:
    def __init__(self, model, processor, device=device):
        self.model = model
        self.processor = processor
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()

    def _get_predictions_and_true_labels(self, data_loader, input_texts_raw, target_diacritics_raw):
        all_pred_indices = []
        all_true_indices = []
        all_input_lengths = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device) # These are padded target indices
                lengths = batch['length'] # Actual lengths of sequences in this batch

                src_padding_mask = (inputs == self.processor.char_to_idx['<PAD>']).to(self.device)
                outputs = self.model(inputs, src_padding_mask=src_padding_mask)
                _, predicted_indices_batch = torch.max(outputs, dim=2)

                all_pred_indices.extend(predicted_indices_batch.cpu().tolist())
                all_true_indices.extend(targets.cpu().tolist()) # Store padded targets
                all_input_lengths.extend(lengths.cpu().tolist())
        
        # Decode and trim to actual length
        true_diac_sequences = []
        pred_diac_sequences = []

        for i in range(len(all_input_lengths)):
            length = all_input_lengths[i]
            
            # True diacritics: use target_diacritics_raw which are not encoded/padded by dataloader
            # This ensures we are comparing against the original, variable-length ground truth
            true_d_seq = target_diacritics_raw[i][:length] 
            true_diac_sequences.append(true_d_seq)
            
            # Predicted diacritics: decode the model's output (which was based on padded input)
            # then trim to the actual length of the original input string
            raw_pred_idx_for_sample = all_pred_indices[i][:length]
            pred_d_seq = self.processor.decode_diacritics(raw_pred_idx_for_sample)
            pred_diac_sequences.append(pred_d_seq)
            
        return true_diac_sequences, pred_diac_sequences


    def character_level_accuracy(self, true_diac_sequences, pred_diac_sequences):
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(true_diac_sequences, pred_diac_sequences):
            for t, p in zip(true_seq, pred_seq):
                if t == p:
                    correct += 1
            total += len(true_seq) # Count all characters in the original sequence length
        return correct / total if total > 0 else 0

    def word_level_accuracy(self, input_texts_raw, true_diac_sequences, pred_diac_sequences):
        correct_words = 0
        total_words = 0
        
        for text_idx, raw_input_text in enumerate(input_texts_raw):
            true_seq = true_diac_sequences[text_idx]
            pred_seq = pred_diac_sequences[text_idx]

            # Ensure raw_input_text is split correctly (handle multiple spaces, etc.)
            words = [word for word in raw_input_text.split(' ') if word] # Simple space split
            
            current_char_idx = 0
            if not words: continue

            for word in words:
                word_len = len(word)
                if current_char_idx + word_len > len(true_seq): # Word boundary exceeds sequence length
                    break 

                true_word_diacs = true_seq[current_char_idx : current_char_idx + word_len]
                pred_word_diacs = pred_seq[current_char_idx : current_char_idx + word_len]
                
                if true_word_diacs == pred_word_diacs:
                    correct_words += 1
                
                total_words += 1
                current_char_idx += word_len + 1 # +1 for the space

        return correct_words / total_words if total_words > 0 else 0

    def diacritic_error_rate(self, true_diac_sequences, pred_diac_sequences):
        errors = 0
        total_diacritics_in_true = 0 # Number of actual diacritics in the ground truth
        
        for true_seq, pred_seq in zip(true_diac_sequences, pred_diac_sequences):
            for t_diac, p_diac in zip(true_seq, pred_seq):
                is_true_diac_present = (t_diac != '' and t_diac != self.processor.idx_to_diacritic.get(self.processor.pad_diacritic_idx,''))
                
                if is_true_diac_present:
                    total_diacritics_in_true +=1
                    if t_diac != p_diac:
                        errors +=1
                elif t_diac != p_diac: # False positive: predicted a diacritic where there was none
                    errors +=1 
                    # Note: Standard DER usually focuses on errors on actual diacritics. 
                    # This alternative counts false positives too. For strict DER, remove this else block.
        
        return errors / total_diacritics_in_true if total_diacritics_in_true > 0 else float('inf') if errors > 0 else 0


    def evaluate(self, test_loader, test_inputs_raw, test_targets_raw): # Pass raw texts for accurate eval
        true_diac_sequences, pred_diac_sequences = self._get_predictions_and_true_labels(test_loader, test_inputs_raw, test_targets_raw)
        
        char_accuracy = self.character_level_accuracy(true_diac_sequences, pred_diac_sequences)
        word_accuracy = self.word_level_accuracy(test_inputs_raw, true_diac_sequences, pred_diac_sequences) # Use raw inputs
        der = self.diacritic_error_rate(true_diac_sequences, pred_diac_sequences)
        
        print(f"\n--- Evaluation Metrics ---")
        print(f"Character-Level Accuracy (CLA): {char_accuracy:.4f}")
        print(f"Word Error Rate (WER) - based on full diacritization: {1 - word_accuracy:.4f} (Word Accuracy: {word_accuracy:.4f})")
        print(f"Diacritic Error Rate (DER): {der:.4f}")
        
        self.show_example_predictions(test_inputs_raw[:5], true_diac_sequences[:5], pred_diac_sequences[:5])
        
        return {
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'diacritic_error_rate': der
        }

    def show_example_predictions(self, inputs_raw, true_diac_sequences, pred_diac_sequences, num_examples=5):
        print("\n--- Example Predictions ---")
        for i in range(min(num_examples, len(inputs_raw))):
            input_text = inputs_raw[i]
            true_diacs = true_diac_sequences[i]
            pred_diacs = pred_diac_sequences[i]
            
            # Ensure lengths match for apply_diacritics, trim if necessary
            min_len = min(len(input_text), len(true_diacs), len(pred_diacs))

            true_text_display = self.processor.apply_diacritics(input_text[:min_len], true_diacs[:min_len])
            pred_text_display = self.processor.apply_diacritics(input_text[:min_len], pred_diacs[:min_len])
            
            print(f"\nExample {i+1}:")
            print(f"Input:      {input_text}")
            print(f"True:       {true_text_display}")
            print(f"Predicted:  {pred_text_display}")
            
            # Accuracy for this example
            correct_diacs = sum(1 for t, p in zip(true_diacs[:min_len], pred_diacs[:min_len]) if t == p)
            example_acc = correct_diacs / min_len if min_len > 0 else 0
            print(f"Accuracy (this example): {example_acc:.2f}")
        print("-" * 30)


class ArabicDiacritizer:
    def __init__(self, model_path, device=device):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load processor attributes from checkpoint
        self.processor = ArabicDiacriticsDataProcessor(max_sequence_length=checkpoint['max_sequence_length'])
        self.processor.char_to_idx = checkpoint['char_to_idx']
        self.processor.idx_to_char = checkpoint['idx_to_char']
        self.processor.diacritic_to_idx = checkpoint['diacritic_to_idx']
        self.processor.idx_to_diacritic = checkpoint['idx_to_diacritic']
        
        # Infer model params from checkpoint if available, or use defaults
        # This part assumes the model was an AdvancedArabicDiacritizationModel
        # For more robustness, model class and its hyperparams should also be saved.
        self.model = AdvancedArabicDiacritizationModel(
            vocab_size=len(self.processor.char_to_idx),
            diacritic_size=len(self.processor.diacritic_to_idx),
            max_sequence_length=self.processor.max_sequence_length
            # Add other params like d_model, nhead if saved in checkpoint, otherwise defaults are used
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path} and moved to {self.device}")

    def diacritize_text(self, text):
        if not text.strip():
            return ""
            
        # Preprocess (normalize, strip existing diacritics for a clean input)
        # It's important that the input text for `encode_input` is non-diacritized
        # and matches the way training data was prepared.
        input_text_clean = self.processor.strip_diacritics(unicodedata.normalize('NFC', text.strip()))
        if not input_text_clean: # If stripping all leaves empty string
             return text # Return original if it becomes empty, or handle as error

        encoded_input = self.processor.encode_input(input_text_clean, pad=True) # Always pad for model input
        input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(self.device)
        
        src_padding_mask = (input_tensor == self.processor.char_to_idx['<PAD>']).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, src_padding_mask=src_padding_mask)
            _, pred_indices = torch.max(outputs, dim=2)
        
        # Decode diacritics, considering the actual length of the *cleaned* input text
        actual_length = len(input_text_clean) # Length before padding
        pred_diacritics_decoded = self.processor.decode_diacritics(pred_indices[0][:actual_length].cpu().tolist())
        
        diacritized_text = self.processor.apply_diacritics(input_text_clean, pred_diacritics_decoded)
        return diacritized_text

    def batch_diacritize(self, texts):
        return [self.diacritize_text(text) for text in texts]


def load_csv_data_from_directory(directory_path, column_name='text_with_harakat', sample_size=None):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files found in directory: {directory_path}")
        return []
    
    all_samples = []
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(csv_file, usecols=[column_name]) # Only load the necessary column
            df.dropna(subset=[column_name], inplace=True)
            all_samples.extend(df[column_name].tolist())
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")

    if sample_size and sample_size < len(all_samples):
        print(f"Sampling {sample_size} instances from {len(all_samples)} total.")
        all_samples = random.sample(all_samples, sample_size)
    
    print(f"Loaded {len(all_samples)} text samples.")
    return all_samples


def plot_training_history(history, model_name, results_dir):
    if not history['val_loss']: # Only plot train loss if no validation
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title(f'Training Loss for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{model_name}_train_loss.png")
        plt.close()
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, linestyle='--', label='Train Loss')
    ax1.plot(history['val_loss'], color=color, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(history['val_accuracy'], color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(f'Training History for {model_name}')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{results_dir}/{model_name}_training_history.png")
    plt.close()
    print(f"Training history plot saved to {results_dir}/{model_name}_training_history.png")


def train_model_main_flow(
    # Data params
    data_directory="data/", # Expects a 'data' folder with CSVs
    sample_data_size=None, # Use a small subset for quick tests, e.g., 10000
    # Model HPs (Transformer specific)
    max_sequence_length=MAX_SEQUENCE_LENGTH, 
    d_model=EMBEDDING_DIM,
    nhead=NUM_ATTENTION_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=HIDDEN_DIM, # For FFN in Transformer
    # Training HPs
    batch_size=BATCH_SIZE, 
    dropout_rate=DROPOUT_RATE, 
    learning_rate=LEARNING_RATE, 
    epochs=EPOCHS,
    patience_epochs=PATIENCE, # Renamed from 'patience'
    weight_decay=1e-5,
    # Paths
    results_dir="results/",
    run_name_prefix="AdvArabicDiac" # Advanced Arabic Diacritization
):
    model_name = (f"{run_name_prefix}_Seq{max_sequence_length}_D{d_model}_H{nhead}_L{num_encoder_layers}"
                  f"_FFN{dim_feedforward}_B{batch_size}_LR{learning_rate:.0e}_Drop{dropout_rate}")
    
    print(f"\n{'='*30}\nTraining model: {model_name}\n{'='*30}")
    
    # Create directories for this specific run
    run_results_dir = os.path.join(results_dir, model_name)
    checkpoints_dir = os.path.join(run_results_dir, "checkpoints")
    final_model_dir = os.path.join(run_results_dir, "model")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}_best.pt")
    # Processor will be saved implicitly with the best model checkpoint for this setup
    final_model_path = os.path.join(final_model_dir, f"{model_name}_final.pt") # This will be same as best after training
    
    print("1. Loading data...")
    diacritized_texts = load_csv_data_from_directory(data_directory, sample_size=sample_data_size)
    if not diacritized_texts:
        print("No data loaded. Exiting.")
        return None

    print("2. Initializing data processor...")
    data_processor = ArabicDiacriticsDataProcessor(max_sequence_length=max_sequence_length)
    
    print("3. Preparing dataset (this may take a while)...")
    inputs, targets = data_processor.prepare_dataset(diacritized_texts)
    if not inputs:
        print("Dataset preparation resulted in no usable data. Exiting.")
        return None

    print(f"Vocabulary size: {len(data_processor.char_to_idx)}")
    print(f"Number of diacritic classes: {len(data_processor.diacritic_to_idx)}")

    # print("Applying data augmentation...") # Augmentation is light for now
    # augmented_inputs, augmented_targets = data_processor.data_augmentation(inputs, targets, augmentation_factor=0.05) # Low factor

    # Using original inputs/targets, augmentation can be added if beneficial
    augmented_inputs, augmented_targets = inputs, targets

    print("4. Splitting dataset...")
    # Train: 70%, Val: 15%, Test: 15%
    inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
        augmented_inputs, augmented_targets, test_size=0.15, random_state=42, shuffle=True
    )
    val_relative_size = 0.15 / (1 - 0.15) # = 0.15 / 0.85
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs_train_val, targets_train_val, test_size=val_relative_size, random_state=42, shuffle=True
    )
    
    print(f"Train set: {len(inputs_train)} samples")
    print(f"Validation set: {len(inputs_val)} samples")
    print(f"Test set: {len(inputs_test)} samples")

    if not inputs_train or not inputs_val or not inputs_test:
        print("One of the data splits is empty. Check data size and split ratios. Exiting.")
        return None

    print("5. Creating PyTorch datasets and dataloaders...")
    train_dataset = ArabicDiacriticsDataset(inputs_train, targets_train, data_processor)
    val_dataset = ArabicDiacriticsDataset(inputs_val, targets_val, data_processor)
    test_dataset = ArabicDiacriticsDataset(inputs_test, targets_test, data_processor)
    
    # Consider num_workers for faster data loading if not on Windows or if multiprocessing is stable
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("6. Initializing Advanced Transformer Model...")
    model = AdvancedArabicDiacritizationModel(
        vocab_size=len(data_processor.char_to_idx),
        diacritic_size=len(data_processor.diacritic_to_idx),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout_rate,
        max_sequence_length=max_sequence_length
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    print("7. Training model...")
    training_start_time = time.time()
    trainer = DiacritizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    
    # The train method now returns the model and history
    model, training_history = trainer.train(
        epochs=epochs,
        checkpoint_path=checkpoint_path, # Best model saved here
        patience_epochs=patience_epochs
    )
    training_time_seconds = time.time() - training_start_time
    print(f"Training completed in {training_time_seconds/60:.2f} minutes.")

    # Plot training history
    plot_training_history(training_history, model_name, run_results_dir)

    print("8. Evaluating model on the test set...")
    # Load the best model for evaluation
    if os.path.exists(checkpoint_path):
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint.get('epoch', 'N/A')} for final evaluation.")
    else:
        print("Warning: No best model checkpoint found. Evaluating with the current model state.")

    evaluator = DiacritizationEvaluator(model, data_processor, device)
    inference_start_time = time.time()
    # Pass raw test inputs/targets for correct WER/DER calculation without padding influence
    metrics = evaluator.evaluate(test_loader, inputs_test, targets_test) 
    inference_time_seconds = time.time() - inference_start_time
    print(f"Evaluation completed in {inference_time_seconds:.2f} seconds.")

    # 9. Save final (best) model information more comprehensively
    # The checkpoint_path already stores the best model. We can copy it or add more info.
    # For consistency, final_model_path will be the same as checkpoint_path after training.
    if os.path.exists(checkpoint_path):
        os.rename(checkpoint_path, final_model_path) # Or copy
        print(f"Best model saved as final model: {final_model_path}")


    # 10. Save results summary
    results_summary = {
        'model_name': model_name,
        'hyperparameters': {
            'max_sequence_length': max_sequence_length, 'd_model': d_model,
            'nhead': nhead, 'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward, 'batch_size': batch_size,
            'dropout_rate': dropout_rate, 'learning_rate': learning_rate,
            'epochs_run': len(training_history['train_loss']), 'target_epochs': epochs, 
            'patience': patience_epochs, 'weight_decay': weight_decay
        },
        'metrics': metrics,
        'trainable_parameters': total_params,
        'training_time_minutes': round(training_time_seconds / 60, 2),
        'inference_time_seconds_on_test_set': round(inference_time_seconds, 2),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'final_model_path': final_model_path,
        'training_history_plot': f"{run_results_dir}/{model_name}_training_history.png"
    }
    
    results_json_path = os.path.join(run_results_dir, f"{model_name}_results.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    print(f"Results summary saved to {results_json_path}")
    
    return results_summary


def run_hyperparameter_search_advanced(
    base_results_dir="results_hyperparam_search_advanced/",
    num_random_configs=5 # Number of random configurations to try
):
    """Run hyperparameter optimization with multiple combinations using the advanced model."""
    
    # Define hyperparameter search space (more focused for Transformer)
    param_search_space = {
        'max_sequence_length': [128, 256], # Shorter sequences are faster to train
        'd_model': [128, 256], # Model dimension
        'nhead': [4, 8], # Number of attention heads (must be divisor of d_model)
        'num_encoder_layers': [2, 4], # Depth of the transformer
        'dim_feedforward': [512, 1024], # FFN hidden size
        'batch_size': [32, 64],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [1e-4, 5e-5, 2e-5],
        'weight_decay': [1e-5, 1e-6],
        'epochs': [EPOCHS], # Fixed, rely on early stopping
        'patience_epochs': [PATIENCE], # Fixed
        'sample_data_size': [20000] # Use a consistent, reasonably sized sample for HP search for speed
                                    # Set to None to use all data, but search will be very slow.
    }
    
    all_configs = []
    # Ensure nhead is compatible with d_model
    for _ in range(num_random_configs):
        config = {}
        for key, values in param_search_space.items():
            config[key] = random.choice(values)
        
        # Constraint: nhead must be a divisor of d_model
        while config['d_model'] % config['nhead'] != 0:
            config['nhead'] = random.choice(param_search_space['nhead']) 
            # Or, more systematically:
            # possible_nheads = [h for h in param_search_space['nhead'] if config['d_model'] % h == 0]
            # if possible_nheads: config['nhead'] = random.choice(possible_nheads)
            # else: # d_model and nhead combination is impossible from grid, regenerate d_model or skip
            #   config['d_model'] = random.choice(param_search_space['d_model']) # try regenerating d_model

        all_configs.append(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_search_results_dir = os.path.join(base_results_dir, f"search_{timestamp}")
    os.makedirs(current_search_results_dir, exist_ok=True)
    
    # Save parameter grid for reference
    with open(os.path.join(current_search_results_dir, "param_search_space.json"), 'w') as f:
        json.dump(param_search_space, f, indent=4)
    with open(os.path.join(current_search_results_dir, "generated_configs.json"), 'w') as f:
        json.dump(all_configs, f, indent=4)

    all_run_results = []
    print(f"Starting hyperparameter search with {len(all_configs)} random configurations.")
    
    for i, params_combo in enumerate(all_configs):
        print(f"\n--- Running Configuration {i+1}/{len(all_configs)} ---")
        print(params_combo)
        
        try:
            # Pass data_directory and results_dir correctly
            run_result = train_model_main_flow(
                data_directory="data/", # Or make this configurable
                results_dir=current_search_results_dir, # Each run will have its subfolder within this
                run_name_prefix=f"Config{i+1}",
                **params_combo
            )
            if run_result:
                all_run_results.append(run_result)
        except Exception as e:
            print(f"Error training model with parameters {params_combo}: {str(e)}")
            # Log the error
            with open(os.path.join(current_search_results_dir, "errors.log"), 'a') as f:
                f.write(f"Timestamp: {datetime.now()}\nParams: {params_combo}\nError: {str(e)}\n\n")
    
    # Save all results summary
    with open(os.path.join(current_search_results_dir, "all_runs_summary.json"), 'w') as f:
        json.dump(all_run_results, f, indent=4)
    
    # Create comparison report (if any results were generated)
    if all_run_results:
        create_comparison_report_advanced(all_run_results, current_search_results_dir)
    else:
        print("No results generated from hyperparameter search.")
        
    return all_run_results


def create_comparison_report_advanced(results_list, output_dir_path):
    if not results_list:
        print("No results to compare.")
        return

    comparison_data = []
    for res in results_list:
        if res and 'metrics' in res and 'hyperparameters' in res: # Check for valid result structure
            flat_data = {
                'model_name': res.get('model_name', 'N/A'),
                'char_accuracy': res['metrics'].get('character_accuracy', 0),
                'word_accuracy': res['metrics'].get('word_accuracy', 0),
                'der': res['metrics'].get('diacritic_error_rate', float('inf')),
                'params': res.get('trainable_parameters', 0),
                'train_time_min': res.get('training_time_minutes', 0),
            }
            flat_data.update(res['hyperparameters']) # Add all hyperparams
            comparison_data.append(flat_data)
        else:
            print(f"Skipping invalid result structure: {res}")


    if not comparison_data:
        print("No valid result data to create a comparison report.")
        return

    df = pd.DataFrame(comparison_data)
    df_sorted = df.sort_values(by=['char_accuracy', 'der'], ascending=[False, True]) # Sort by CLA (desc) then DER (asc)
    
    csv_path = os.path.join(output_dir_path, "hyperparam_comparison_summary.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"Comparison report saved to {csv_path}")

    # Visualizations (can be expanded)
    if len(df_sorted) > 1:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_sorted, x='params', y='char_accuracy', hue='num_encoder_layers', size='d_model')
        plt.title('Character Accuracy vs. Model Size (Params)')
        plt.xlabel('Number of Trainable Parameters')
        plt.ylabel('Character Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_path, "accuracy_vs_model_size.png"))
        plt.close()

        # Top 5 models bar chart
        top_n = min(5, len(df_sorted))
        top_models_df = df_sorted.head(top_n)
        plt.figure(figsize=(12, 7))
        bars = plt.barh(top_models_df['model_name'], top_models_df['char_accuracy'], color=sns.color_palette("viridis", top_n))
        plt.xlabel('Character Accuracy')
        plt.title(f'Top {top_n} Models by Character Accuracy')
        plt.gca().invert_yaxis() # Display best on top
        for bar in bars:
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                     f'{bar.get_width():.4f}', va='center', ha='left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_path, "top_models_bar_chart.png"))
        plt.close()

    print(f"Visualizations saved in {output_dir_path}")
    return df_sorted


def main():
    print("=== Advanced Arabic Diacritization Model Training & Evaluation ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {device}")

    # OPTION 1: Train a single model with specific (or default) hyperparameters
    # print("\n--- Training a Single Model ---")
    # single_run_results = train_model_main_flow(
    #     data_directory="data/",       # Make sure your data is in ./data/
    #     sample_data_size=5000,        # Use a small sample for a quick test, None for full data
    #     max_sequence_length=128,
    #     d_model=128,
    #     nhead=4,
    #     num_encoder_layers=3,
    #     dim_feedforward=512,
    #     batch_size=64,
    #     dropout_rate=0.1,
    #     learning_rate=LEARNING_RATE, # Use the default LEARNING_RATE or specify
    #     epochs=10,                 # Low epochs for a quick test
    #     patience_epochs=3,
    #     results_dir="results_single_run/"
    # )
    # if single_run_results:
    #     print("\nSingle run completed. Metrics:")
    #     print(json.dumps(single_run_results.get('metrics', {}), indent=2))
    #     # To use the trained model:
    #     # diacritizer = ArabicDiacritizer(model_path=single_run_results['final_model_path'])
    #     # print(diacritizer.diacritize_text("سلام عليكم"))


    # OPTION 2: Run hyperparameter search
    print("\n--- Running Hyperparameter Search ---")
    # This will take a long time if sample_data_size is large or None
    # and num_random_configs is high.
    search_results = run_hyperparameter_search_advanced(
         base_results_dir="results_hyperparam_search_advanced/",
         num_random_configs=3 # Try 3 random configs for a demo. Increase for real search.
    )
    if search_results:
        print("\nHyperparameter search completed.")
        # Find the best model from the search
        best_run = max(search_results, key=lambda r: r['metrics']['character_accuracy'] if r and 'metrics' in r and 'character_accuracy' in r['metrics'] else -1)
        if best_run and 'final_model_path' in best_run:
            print(f"\nBest model from search: {best_run['model_name']}")
            print(f"Path: {best_run['final_model_path']}")
            print(f"Metrics: {json.dumps(best_run['metrics'], indent=2)}")
            
            # Example of using the best model
            # try:
            #     best_diacritizer = ArabicDiacritizer(model_path=best_run['final_model_path'])
            #     test_sentence = "مرحبا بالعالم"
            #     diacritized_sentence = best_diacritizer.diacritize_text(test_sentence)
            #     print(f"Test: '{test_sentence}' -> '{diacritized_sentence}'")
            # except Exception as e:
            #     print(f"Could not load or use the best model due to error: {e}")
        else:
            print("Could not determine the best model from the search results.")
    else:
        print("Hyperparameter search did not produce any results.")

if __name__ == "__main__":
    # Ensure 'data' directory exists.
    # For a quick test, you might want to create a 'data' folder and put a small CSV file in it.
    # Example CSV content (e.g., data/sample.csv):
    # text_with_harakat
    # السَّلَامُ عَلَيْكُمْ
    # كَيْفَ حَالُكَ؟
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please put your CSV dataset(s) there.")
        # Create a dummy CSV for testing if 'data' is empty
        if not glob.glob(os.path.join("data", "*.csv")):
            dummy_data = pd.DataFrame({
                'text_with_harakat': [
                    "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ.",
                    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ.",
                    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ.",
                    "اللَّهُمَّ صَلِّ عَلَى مُحَمَّدٍ وَآلِ مُحَمَّدٍ.",
                    "هَٰذَا نَصٌّ تَجْرِيبِيٌّ بِالتَّشْكِيلِ الْكَامِلِ."
                ]*200 # Make it a bit larger for splits
            })
            dummy_csv_path = os.path.join("data", "dummy_dataset.csv")
            dummy_data.to_csv(dummy_csv_path, index=False, encoding='utf-8')
            print(f"Created a dummy dataset at {dummy_csv_path} for demonstration.")


    main()