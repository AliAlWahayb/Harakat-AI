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


# Constants
MAX_SEQUENCE_LENGTH = 256  # Default, can be overridden by hyperparameter tuning
BATCH_SIZE = 64           # Default, can be overridden by hyperparameter tuning
EMBEDDING_DIM = 256       # Default, can be overridden by hyperparameter tuning
HIDDEN_DIM = 512          # Used for LSTM in original, Transformer uses dim_feedforward
DROPOUT_RATE = 0.5        # Default, can be overridden by hyperparameter tuning
LEARNING_RATE = 1e-3      # Default, can be overridden by hyperparameter tuning
EPOCHS = 500              # Default, can be overridden by hyperparameter tuning

# Transformer Specific Defaults (can be added to hyperparameter tuning)
NHEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 6 # Number of Transformer encoder layers
DIM_FEEDFORWARD = 2048 # Dimension of the feedforward network model in Transformer

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
        
        # Define Arabic diacritics
        self.diacritics = [
            '\u064B',  # Fathatan
            '\u064C',  # Dammatan
            '\u064D',  # Kasratan
            '\u064E',  # Fatha
            '\u064F',  # Damma
            '\u0650',  # Kasra
            '\u0651',  # Shadda
            '\u0652',  # Sukun
            # Removed less common/combined diacritics for simplicity, can be added back
            # '\u0653',  # Maddah 
            # '\u0654',  # Hamza above
            # '\u0655',  # Hamza below
            # '\u0670',  # Superscript Alef
            ''          # No diacritic
        ]
        
        # Initialize diacritic mappings
        for i, diac in enumerate(self.diacritics):
            self.diacritic_to_idx[diac] = i
            self.idx_to_diacritic[i] = diac
            
    def get_diacritic_size(self):
        return len(self.diacritics)

    def get_vocab_size(self):
        return len(self.char_to_idx)

    def strip_diacritics(self, text):
        """Remove diacritics from Arabic text while preserving characters"""
        # More robust stripping using unicodedata
        normalized_text = unicodedata.normalize('NFD', text)
        return ''.join([c for c in normalized_text if unicodedata.category(c) != 'Mn' and c not in self.diacritics])

    def extract_diacritics_from_char(self, char_with_diacritics):
        """Extracts diacritics following a base character."""
        base_char = ''
        diacs = []
        temp_diacs = [] # To handle multi-part diacritics like Shadda+Fatha

        # First, isolate the base character
        for c in char_with_diacritics:
            if c not in self.diacritics and unicodedata.category(c) != 'Mn':
                base_char += c # Should ideally be one character
            else:
                if c in self.diacritics: # Only known diacritics
                    temp_diacs.append(c)
        
        if not base_char and char_with_diacritics and char_with_diacritics[0] not in self.diacritics:
             base_char = char_with_diacritics[0]


        # Logic for combining shadda with other diacritics (can be expanded)
        if '\u0651' in temp_diacs: # Shadda
            shadda_present = True
            other_diacs = [d for d in temp_diacs if d != '\u0651']
            if other_diacs: # Shadda + another diacritic
                # Prioritize simple diacritics (Fatha, Damma, Kasra) with Shadda
                # This is a simplification. Real rules are complex.
                # For this model, we'll treat them as separate predictions or map to combined symbols if they exist in `self.diacritics`
                # Current setup: predict one diacritic per character. Shadda can be one, Fatha another.
                # A better approach might be multi-label classification or sequence tagging for multiple diacritics.
                # For now, we'll just pick the first non-shadda if present, or shadda if alone, or empty.
                # This part might need refinement based on how combined diacritics are handled.
                # The current `self.diacritics` list implies one diacritic per character position.
                # We will map shadda if it exists, otherwise the other diacritic.
                # If the user wants to handle shadda+vowel as a single token, `self.diacritics` must be updated.
                if '\u0651' in self.diacritic_to_idx:
                    diacs.append('\u0651') # Predict Shadda
                if other_diacs and other_diacs[0] in self.diacritic_to_idx:
                     pass # In a multi-label setup, we would add other_diacs[0] too.
                          # For single label, this is problematic. The current user code implies single label.
                          # Let's assume the target `diacritics` list should match `input_text` length.
                          # This means each input character gets one diacritic target.
                          # The original code's `extract_diacritics` seems to handle this by joining.
                          # Let's stick to a simplified model where one primary diacritic is chosen.

            else: # Shadda alone
                 if '\u0651' in self.diacritic_to_idx:
                    diacs.append('\u0651')
        elif temp_diacs: # Other diacritics without Shadda
            if temp_diacs[0] in self.diacritic_to_idx:
                diacs.append(temp_diacs[0])

        return base_char, ''.join(diacs) if diacs else ''


    def create_character_mappings(self, texts):
        """Create character to index mappings from a list of texts"""
        unique_chars = set()
        for text in texts:
            stripped_text = self.strip_diacritics(text) # Ensure we map only base characters
            for char in stripped_text:
                unique_chars.add(char)
        
        # Create mappings
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(list(unique_chars)))}
        self.char_to_idx['<PAD>'] = 0  # Add padding token
        self.char_to_idx['<UNK>'] = len(self.char_to_idx) # Add UNK token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        print(f"Vocabulary size (chars): {len(self.char_to_idx)}")
        print(f"Diacritic set size: {len(self.diacritic_to_idx)}")

    def prepare_dataset(self, diacritized_texts):
        """Prepare dataset from diacritized texts"""
        inputs = []
        targets = []
        
        # temp_texts = []
        # for text in diacritized_texts:
        #     # Normalize and clean text slightly
        #     text = unicodedata.normalize('NFC', text) # Compose combined characters
        #     text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        #     if text:
        #         temp_texts.append(text)
        # diacritized_texts = temp_texts

        # First pass: build character vocabulary from stripped text
        pre_stripped_texts = [self.strip_diacritics(text) for text in diacritized_texts]
        self.create_character_mappings(pre_stripped_texts)

        for text in tqdm(diacritized_texts, desc="Processing texts for dataset"):
            if not text: continue

            input_chars = []
            diacritics_for_chars = []
            
            current_char_buffer = ""
            for char_code in text: # Iterate through each Unicode character code point
                # char = chr(char_code) if isinstance(char_code, int) else char_code
                char = char_code

                if char not in self.diacritics and unicodedata.category(char) != 'Mn': # It's a base character
                    if current_char_buffer: # Process previous character and its diacritics
                        base, diac = self.extract_diacritics_from_char(current_char_buffer)
                        if base: # Only add if a base character was found
                            input_chars.append(base)
                            diacritics_for_chars.append(diac)
                    current_char_buffer = char # Start new buffer with current base character
                else: # It's a diacritic or part of one
                    current_char_buffer += char
            
            # Process the last character buffer
            if current_char_buffer:
                base, diac = self.extract_diacritics_from_char(current_char_buffer)
                if base:
                    input_chars.append(base)
                    diacritics_for_chars.append(diac)

            input_text = "".join(input_chars)

            if len(input_text) != len(diacritics_for_chars):
                # This can happen if the logic for splitting char/diacritics is imperfect
                # or if text contains only diacritics, etc.
                # print(f"Warning: Mismatch in lengths for text: '{text}'. Input: {len(input_text)}, Diacritics: {len(diacritics_for_chars)}. Skipping.")
                # print(f"Original: {text}, Processed Input: {input_text}, Processed Diacritics: {diacritics_for_chars}")
                continue # Skip problematic examples

            if input_text: # Only add if we have a valid input string
                inputs.append(input_text)
                targets.append(diacritics_for_chars)
        
        print(f"Prepared {len(inputs)} input sequences and {len(targets)} target sequences.")
        return inputs, targets

    def encode_input(self, text, pad=True):
        """Encode input text to integer sequence"""
        if len(text) > self.max_sequence_length:
            text = text[:self.max_sequence_length]
        
        encoded = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        
        if pad and len(encoded) < self.max_sequence_length:
            encoded = encoded + [self.char_to_idx['<PAD>']] * (self.max_sequence_length - len(encoded))
        
        return encoded
    
    def encode_target(self, diacritics_list, pad=True):
        """Encode target diacritics to integer sequence"""
        # diacritics_list is a list of diacritic strings, one for each character
        if len(diacritics_list) > self.max_sequence_length:
            diacritics_list = diacritics_list[:self.max_sequence_length]
        
        # Default to "no diacritic" index
        no_diacritic_idx = self.diacritic_to_idx.get("", -1) # Assuming "" is "no diacritic"
        if no_diacritic_idx == -1:
             # Fallback if "" not in map, though it should be.
             # This indicates a problem with diacritic_to_idx initialization.
             print("Warning: '' (no diacritic) not found in diacritic_to_idx. Using 0 as fallback.")
             no_diacritic_idx = 0


        encoded = [self.diacritic_to_idx.get(diac, no_diacritic_idx) for diac in diacritics_list]
        
        if pad and len(encoded) < self.max_sequence_length:
            # Pad with "no diacritic" index. Original code used this, which is fine.
            # Or, pad with a specific ignore_index for the loss function if different from "no diacritic".
            # The CrossEntropyLoss ignore_index should match this padding value if it's not a valid target class.
            # If 0 is <PAD> for inputs, and we use ignore_index=0 for loss, this might be fine if 0 is also PAD for targets.
            # The user's original code used self.diacritic_to_idx[""] for padding targets, let's stick to that.
            encoded = encoded + [no_diacritic_idx] * (self.max_sequence_length - len(encoded))
        
        return encoded
    
    def decode_diacritics(self, indices):
        """Convert diacritic indices back to diacritics"""
        # Handle potential out-of-bound indices if model predicts something unexpected
        return [self.idx_to_diacritic.get(idx, '') for idx in indices]

    def apply_diacritics(self, text, diacritics_list):
        """Apply diacritics (list of strings) to text (string)"""
        result = []
        # Ensure text and diacritics_list are of the same length for zip
        min_len = min(len(text), len(diacritics_list))
        for i in range(min_len):
            result.append(text[i])
            result.append(diacritics_list[i])
        # If text is longer, append remaining characters
        if len(text) > min_len:
            result.append(text[min_len:])
        return "".join(result)

    def data_augmentation(self, inputs, targets, augmentation_factor=0.1): # Reduced default factor
        """Apply data augmentation specific to Arabic text"""
        augmented_inputs = []
        augmented_targets = []
        
        for input_text, target_diacritics_list in zip(inputs, targets):
            augmented_inputs.append(input_text)
            augmented_targets.append(target_diacritics_list)
            
            if random.random() > augmentation_factor:
                continue
            
            # 1. Character substitution (similar looking characters)
            # This needs to be done carefully to not break the char_to_idx mapping too much
            # Or, ensure substituted chars are in vocab. For now, let's keep it simple.
            if len(input_text) > 5 and random.random() < 0.3: # Reduced probability
                char_map = {'ا': 'أ', 'أ': 'ا', 'إ': 'ا', 'ه': 'ة', 'ة': 'ه', 'ي': 'ى', 'ى': 'ي'}
                pos = random.randint(0, len(input_text) - 1)
                original_char = input_text[pos]
                if original_char in char_map:
                    substituted_char = char_map[original_char]
                    # Ensure substituted char is in vocabulary
                    if substituted_char in self.char_to_idx:
                        new_input_text_list = list(input_text)
                        new_input_text_list[pos] = substituted_char
                        augmented_inputs.append("".join(new_input_text_list))
                        augmented_targets.append(target_diacritics_list) # Diacritics remain the same for the position
            
            # 2. Random character deletion (simulates typos) - use with caution
            if len(input_text) > 10 and random.random() < 0.2: # Reduced probability
                pos = random.randint(0, len(input_text) - 1)
                augmented_input = input_text[:pos] + input_text[pos+1:]
                augmented_target = target_diacritics_list[:pos] + target_diacritics_list[pos+1:]
                if augmented_input: # Ensure not empty
                    augmented_inputs.append(augmented_input)
                    augmented_targets.append(augmented_target)
        
        return augmented_inputs, augmented_targets


# PyTorch Dataset
class ArabicDiacriticsDataset(Dataset):
    def __init__(self, inputs, targets, processor):
        self.inputs = inputs
        self.targets = targets # This is a list of lists of diacritic strings
        self.processor = processor
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_diacritics_list = self.targets[idx] # List of diacritic strings
        
        encoded_input = self.processor.encode_input(input_text, pad=True)
        encoded_target = self.processor.encode_target(target_diacritics_list, pad=True)
        
        return {
            'input': torch.tensor(encoded_input, dtype=torch.long),
            'target': torch.tensor(encoded_target, dtype=torch.long),
            'length': min(len(input_text), self.processor.max_sequence_length) # Actual length before padding
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Advanced PyTorch Model with CNN + Transformer Encoder
class AdvancedArabicDiacritizationModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        diacritic_size, 
        max_sequence_length=MAX_SEQUENCE_LENGTH, # Used for positional encoding max_len
        embedding_dim=EMBEDDING_DIM,
        # CNN parameters (can be tuned)
        cnn_out_channels=256, # Output channels for each CNN kernel size set
        cnn_kernel_sizes=[3, 5, 7], # List of kernel sizes
        # Transformer parameters
        d_model=EMBEDDING_DIM, # Input feature dimension for Transformer, should match embedding_dim or CNN output
        nhead=NHEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        transformer_dropout=0.1, # Dropout for transformer layers
        dropout_rate=DROPOUT_RATE # General dropout for other layers if needed
    ):
        super(AdvancedArabicDiacritizationModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.diacritic_size = diacritic_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model # This will be the input dim to Transformer

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0 # Assuming 0 is PAD_IDX in char_to_idx
        )
        
        # Positional Encoding for Transformer
        self.pos_encoder = PositionalEncoding(d_model, dropout=transformer_dropout, max_len=max_sequence_length)

        # 1D CNN layers for local feature extraction
        # The original model concatenated features from different kernel sizes.
        # Let's make the CNN output compatible with d_model for the transformer.
        # Option 1: Each CNN outputs d_model/len(kernel_sizes) and concatenate.
        # Option 2: Each CNN outputs some channels, then a linear layer maps concatenated to d_model.
        # Option 3: Process sequence with one set of CNNs, then a final conv or linear to get d_model.

        # Let's use a simpler CNN structure here: one set of convolutions.
        # Or adapt the user's multi-kernel CNN part.
        # For this iteration, let's keep the multi-kernel idea but ensure output matches d_model.
        # The output of CNNs should be [batch_size, seq_len, d_model]
        
        # For simplicity and robustness, let's adjust the CNN part.
        # If we want to use the user's exact CNN structure, its output (384) needs to be mapped to d_model.
        # Let's assume d_model is the embedding_dim for now.
        # So, CNNs should output features that can be reshaped to [batch_size, seq_len, d_model]
        
        # Simplified CNN: A few layers to process embeddings
        # Or use the user's multi-kernel CNNs and add a linear layer to match d_model
        self.use_cnn_feature_extractor = True # Set to False to bypass CNNs

        if self.use_cnn_feature_extractor:
            # This part needs to be designed carefully.
            # The original CNN output was 384. If d_model is 256, we need a projection.
            # Let's use a single Conv1d layer that directly outputs `d_model` channels.
            # This simplifies the architecture while still allowing CNN to capture local patterns.
            self.conv_extractor = nn.Sequential(
                nn.Conv1d(embedding_dim, d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model // 2), # Added BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model), # Added BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            # If we use this, the input to transformer will be `d_model`
        else:
            # If no CNN, ensure embedding_dim is same as d_model
            if embedding_dim != d_model:
                # This would be an issue, for now assume they are the same if CNN is bypassed
                # Or add a linear projection here: self.input_proj = nn.Linear(embedding_dim, d_model)
                print(f"Warning: Bypassing CNNs, but embedding_dim ({embedding_dim}) != d_model ({d_model}).")
                print("Ensure they match or add a projection layer if bypassing CNNs.")


        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=transformer_dropout,
            batch_first=True # Important: Input shape (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(d_model, diacritic_size) # Transformer output to diacritic vocab
            
    def forward(self, x, src_padding_mask=None): # x shape: [batch_size, seq_len]
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        if self.use_cnn_feature_extractor:
            # CNN expects [batch_size, embedding_dim, seq_len]
            cnn_input = embedded.permute(0, 2, 1) 
            cnn_out = self.conv_extractor(cnn_input) # [batch_size, d_model, seq_len]
            # Transformer expects [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
            # If batch_first=True for TransformerEncoderLayer, then [batch_size, seq_len, d_model]
            transformer_input = cnn_out.permute(0, 2, 1) # [batch_size, seq_len, d_model]
        else:
            # If not using CNNs, the embedded output goes to transformer (assuming embedding_dim == d_model)
            transformer_input = embedded # [batch_size, seq_len, d_model]

        # For TransformerEncoder with batch_first=True, input is [batch, seq, feature]
        # Positional encoding expects [seq, batch, feature] if we use PyTorch's default.
        # Let's adapt PositionalEncoding or transpose.
        # If PositionalEncoding expects [seq, batch, feature], and our current output is [batch, seq, feature]:
        
        # If TransformerEncoder is batch_first=True:
        # Input to self.pos_encoder must be [seq_len, batch_size, d_model]
        # So, permute transformer_input, apply pos_encoder, then permute back.
        # OR, modify PositionalEncoding to accept batch_first.
        # Let's modify PositionalEncoding to handle batch_first=True style input more directly.
        # Simpler: The self.pos_encoder used in official PyTorch tutorials often expects (seq_len, batch, feature).
        # Since our nn.TransformerEncoderLayer is set to batch_first=True, its input is (batch, seq, feature).
        # The self.pos_encoder adds PE assuming (seq_len, batch, feature).
        # To make it simple, we can make the Transformer input (seq_len, batch_size, d_model) before pos_encoder
        # and then permute back if needed, or keep it like that if TransformerEncoder itself is not batch_first.
        # BUT, nn.TransformerEncoderLayer with batch_first=True expects (N, S, E).
        # Let's make PositionalEncoding handle (N,S,E)
        
        # Corrected Positional Encoding application for batch_first=True in Transformer
        # The pos_encoder typically adds to (S, N, E). Our Transformer expects (N, S, E).
        # So, we add PE to `embedded` or `cnn_out_permuted` which are (N,S,E)
        # The `self.pos_encoder.pe` is (max_len, 1, d_model). We need to adapt it for (N,S,E).
        # `self.pos_encoder.pe.squeeze(1)` gives (max_len, d_model)
        # We need to add `pe[:seq_len, :]` to each item in the batch.
        
        # Modified PositionalEncoding class for batch_first scenario
        # For now, let's use the typical Transformer input shape [Seq, Batch, Dim] for PE then switch
        # For batch_first=True in TransformerEncoder:
        # Input: (N, S, E) where N is batch size, S is sequence length, E is feature dimension
        # PositionalEncoding should add PE of shape (S, E) to the features.
        # Standard PositionalEncoding `self.pe` is (max_len, 1, d_model). We slice `[:S, 0, :]`.
        # This `pe_slice` will have shape (S, d_model).
        # `transformer_input` is (N, S, d_model). We can add `pe_slice` to it.
        
        # Simplified Positional Encoding application for batch_first=True
        # Assuming self.pos_encoder.pe is [max_len, 1, d_model]
        seq_len = transformer_input.size(1)
        # pos_encoding_slice shape should be (seq_len, d_model), to be broadcast-added to (N, S, d_model)
        # self.pos_encoder.pe is [max_len, 1, d_model]. Squeeze to [max_len, d_model]
        # Then slice to [seq_len, d_model]
        pe_to_add = self.pos_encoder.pe[:seq_len, :].squeeze(1) # [seq_len, d_model]
        transformer_input = transformer_input + pe_to_add # Broadcasting applies pe_to_add to each batch element
        transformer_input = self.pos_encoder.dropout(transformer_input) # Apply dropout after adding PE

        # Create padding mask for Transformer if provided
        # src_key_padding_mask should be (N, S) where N is batch_size, S is sequence_length
        # True values are positions that will be ignored.
        # If src_padding_mask is not None, it means we have padding from input x (where x == PAD_IDX)

        # Transformer Encoder
        # Input: (N, S, E) if batch_first=True
        # Mask: (N, S) for src_key_padding_mask
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=src_padding_mask) 
        # Output shape: [batch_size, seq_len, d_model]
        
        transformer_output = self.dropout(transformer_output)
        
        # Output layer
        logits = self.fc(transformer_output)  # [batch_size, seq_len, diacritic_size]
        
        return logits


class DiacritizationTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader=None, 
        learning_rate=LEARNING_RATE,
        device=device,
        pad_idx=0 # Assuming 0 is the padding index for targets as well for ignore_index
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        self.pad_idx = pad_idx
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        # Important: ignore_index should be the padding index used in target tensors
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx) 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4) # AdamW is often better
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.2, patience=3, min_lr=1e-7, verbose=True # Made more aggressive
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Create src_key_padding_mask for the Transformer
            # It should be True for padded positions.
            # Assuming input PAD_IDX is 0 (as per processor and model embedding)
            src_padding_mask = (inputs == self.train_loader.dataset.processor.char_to_idx['<PAD>']).to(self.device)

            outputs = self.model(inputs, src_padding_mask=src_padding_mask) # Pass mask to model
            
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes) # Using view is often preferred
            targets_flat = targets.view(-1)
            
            loss = self.criterion(outputs_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        if self.val_loader is None:
            return None, None # Return loss and accuracy
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total_chars = 0 # Total non-padded characters
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                src_padding_mask = (inputs == self.val_loader.dataset.processor.char_to_idx['<PAD>']).to(self.device)
                outputs = self.model(inputs, src_padding_mask=src_padding_mask)
                
                batch_size, seq_len, num_classes = outputs.shape
                outputs_flat = outputs.view(-1, num_classes)
                targets_flat = targets.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=2) # Get predictions per sequence position
                
                # Mask for non-padding target tokens
                # self.pad_idx is the value used for padding in target tensors
                mask = (targets != self.pad_idx) 
                correct += ((predicted == targets) & mask).sum().item()
                total_chars += mask.sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_chars if total_chars > 0 else 0
        
        self.scheduler.step(val_loss) # Step scheduler based on validation loss
        
        return val_loss, accuracy
    
    def train(self, epochs=EPOCHS, checkpoint_path='checkpoints/pytorch_diacritization.pt', patience=10): # Increased patience
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        best_val_accuracy = 0.0 # Track best accuracy for saving model
        patience_counter = 0
        
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, LR: {current_lr:.1e}")
            
            if self.val_loader is not None:
                val_loss, accuracy = self.validate()
                val_losses.append(val_loss)
                val_accuracies.append(accuracy)
                print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
                
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'accuracy': accuracy,
                        'epoch': epoch,
                        'processor_vocab_size': self.model.vocab_size, # Save for model reconstruction
                        'processor_diacritic_size': self.model.diacritic_size # Save for model reconstruction
                    }, checkpoint_path)
                    print(f"Best model (accuracy) saved to {checkpoint_path}")
                else:
                    patience_counter += 1
            else: # No validation loader
                # Save model based on training loss (less ideal)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'epoch': epoch,
                    'processor_vocab_size': self.model.vocab_size,
                    'processor_diacritic_size': self.model.diacritic_size
                }, checkpoint_path)
                print(f"Model saved to {checkpoint_path} (no validation)")

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch duration: {epoch_duration:.2f}s")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} epochs.")
                break
        
        # Load best model if validation was performed
        if self.val_loader is not None and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint.get('epoch', -1)+1} with val accuracy {checkpoint.get('accuracy', 0.0):.4f}")
            except Exception as e:
                print(f"Could not load best model checkpoint: {e}")

        history = {'train_loss': train_losses, 'val_loss': val_losses, 'val_accuracy': val_accuracies}
        return self.model, history


class DiacritizationEvaluator:
    def __init__(self, model, processor, device=device, pad_idx=0):
        self.model = model
        self.processor = processor
        self.device = device
        self.pad_idx = pad_idx # Target padding index
        
        self.model.to(self.device)
        self.model.eval()

    def _get_predictions_and_true_labels(self, data_loader, original_inputs, original_targets_lists):
        all_pred_indices = []
        all_true_indices_unpadded = [] # Unpadded true labels for comparison

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                inputs_tensor = batch['input'].to(self.device)
                # targets_tensor = batch['target'].to(self.device) # Not directly used for prediction list here
                lengths = batch['length'] # Actual sequence lengths

                src_padding_mask = (inputs_tensor == self.processor.char_to_idx['<PAD>']).to(self.device)
                outputs = self.model(inputs_tensor, src_padding_mask=src_padding_mask)
                _, predicted_indices_batch = torch.max(outputs, dim=2) # (batch_size, seq_len)
                
                all_pred_indices.extend(predicted_indices_batch.cpu().tolist())

        # Process true targets to match unpadded length, similar to predictions
        for i in range(len(original_inputs)):
            input_text = original_inputs[i]
            target_diac_list = original_targets_lists[i]
            
            # Important: length should be of the input_text, not the potentially padded original_targets_lists
            actual_len = min(len(input_text), self.processor.max_sequence_length)
            
            # Encode targets without padding to get true indices for the actual length
            true_encoded_unpadded = self.processor.encode_target(target_diac_list[:actual_len], pad=False)
            all_true_indices_unpadded.append(true_encoded_unpadded)
            
            # Also adjust predictions to actual length
            all_pred_indices[i] = all_pred_indices[i][:actual_len]

        return all_pred_indices, all_true_indices_unpadded

    def character_level_accuracy(self, true_diac_indices_list, pred_diac_indices_list):
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(true_diac_indices_list, pred_diac_indices_list):
            for true_idx, pred_idx in zip(true_seq, pred_seq):
                # No need to check for padding here as lists are already unpadded
                if true_idx == pred_idx:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0

    def word_level_accuracy(self, input_texts, true_diac_indices_list, pred_diac_indices_list):
        """ Word Error Rate (WER) / Word Accuracy (WA) based on diacritics """
        correct_words = 0
        total_words = 0

        for i, text in enumerate(input_texts):
            true_diac_indices_for_text = true_diac_indices_list[i]
            pred_diac_indices_for_text = pred_diac_indices_list[i]
            
            words = text.split() # Simple space-based tokenization
            current_char_idx = 0
            
            for word in words:
                word_len = len(word)
                if current_char_idx + word_len > len(true_diac_indices_for_text): # Boundary check
                    # This might happen if text processing (split) differs from sequence generation
                    # print(f"Word boundary issue for text: {text}, word: {word}")
                    continue 

                true_word_diacs = true_diac_indices_for_text[current_char_idx : current_char_idx + word_len]
                pred_word_diacs = pred_diac_indices_for_text[current_char_idx : current_char_idx + word_len]

                if true_word_diacs == pred_word_diacs:
                    correct_words += 1
                total_words += 1
                current_char_idx += word_len + 1 # +1 for the space

        return correct_words / total_words if total_words > 0 else 0

    def diacritic_error_rate(self, true_diac_indices_list, pred_diac_indices_list):
        """ Diacritic Error Rate (DER) - percentage of characters with incorrect diacritics,
            excluding cases where both true and predicted are 'no diacritic' IF no_diacritic_idx is handled.
            More simply: 1 - char_level_accuracy.
            However, standard DER sometimes only counts errors on characters that *should* have a diacritic.
            Let's use 1 - char_level_accuracy for now, as it's a common proxy.
        """
        # The user's previous DER definition was:
        # errors = 0; total = 0
        # for true_seq, pred_seq in zip(true_diacritics, pred_diacritics):
        #     for true_diac, pred_diac in zip(true_seq, pred_seq):
        #         if true_diac != pred_diac and true_diac != '':  # Only count errors on actual diacritics
        #             errors += 1
        #         if true_diac != '':
        #             total += 1
        # Let's implement this with indices.
        errors = 0
        total_actual_diacritics = 0
        no_diac_idx = self.processor.diacritic_to_idx.get("", -1) # Get 'no diacritic' index

        for true_seq, pred_seq in zip(true_diac_indices_list, pred_diac_indices_list):
            for true_idx, pred_idx in zip(true_seq, pred_seq):
                if true_idx != no_diac_idx: # If the character should have a diacritic (is not 'no_diac')
                    total_actual_diacritics +=1
                    if true_idx != pred_idx:
                        errors +=1
        
        return errors / total_actual_diacritics if total_actual_diacritics > 0 else 0


    def evaluate(self, test_loader, test_inputs, test_targets_lists): # test_targets are lists of diacritic strings
        self.model.eval()
        
        pred_indices_list, true_indices_list = self._get_predictions_and_true_labels(
            test_loader, test_inputs, test_targets_lists
        )
        
        char_accuracy = self.character_level_accuracy(true_indices_list, pred_indices_list)
        word_accuracy = self.word_level_accuracy(test_inputs, true_indices_list, pred_indices_list) # Based on diacritic matching per word
        der = self.diacritic_error_rate(true_indices_list, pred_indices_list)
        
        print(f"Character-level accuracy: {char_accuracy:.4f}")
        print(f"Word-level accuracy (diacritics): {word_accuracy:.4f}")
        print(f"Diacritic Error Rate (DER): {der:.4f}")
        
        # Show example predictions (decode indices to diacritics first)
        # Convert true_indices_list and pred_indices_list to lists of diacritic strings
        true_diacritics_str_list = [self.processor.decode_diacritics(seq) for seq in true_indices_list]
        pred_diacritics_str_list = [self.processor.decode_diacritics(seq) for seq in pred_indices_list]

        self.show_example_predictions(test_inputs[:5], true_diacritics_str_list[:5], pred_diacritics_str_list[:5])
        
        return {
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'diacritic_error_rate': der
        }

    def show_example_predictions(self, inputs_texts, true_diacritics_lists, pred_diacritics_lists):
        print("\nExample Predictions:")
        print("-" * 80)
        
        for i, input_text in enumerate(inputs_texts):
            true_diac_list = true_diacritics_lists[i]
            pred_diac_list = pred_diacritics_lists[i]
            
            # Apply diacritics to input text
            true_text_diacritized = self.processor.apply_diacritics(input_text, true_diac_list)
            pred_text_diacritized = self.processor.apply_diacritics(input_text, pred_diac_list)
            
            # Calculate accuracy for this example
            example_correct = sum(t == p for t, p in zip(true_diac_list, pred_diac_list))
            example_accuracy = example_correct / len(true_diac_list) if len(true_diac_list) > 0 else 0

            print(f"Example {i+1}:")
            print(f"Input:      {input_text}")
            print(f"True:       {true_text_diacritized}")
            print(f"Predicted:  {pred_text_diacritized}")
            print(f"Accuracy:   {example_accuracy:.2f}")
            print("-" * 80)


class ArabicDiacritizer:
    def __init__(self, model_path, processor_path, device=device):
        self.device = device
        
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type or pass parameters if saved in checkpoint
        # For AdvancedArabicDiacritizationModel, we need more params.
        # Best to save model hyperparameters in the checkpoint.
        # For now, assuming it's AdvancedArabicDiacritizationModel with default advanced params.
        # This part needs to be robust if you switch between model types or hyperparams.
        
        # Try to load hyperparams from checkpoint if they exist
        model_hyperparams = checkpoint.get('hyperparameters', {})
        vocab_size = checkpoint.get('processor_vocab_size', len(self.processor.char_to_idx))
        diacritic_size = checkpoint.get('processor_diacritic_size', len(self.processor.diacritic_to_idx))

        self.model = AdvancedArabicDiacritizationModel(
            vocab_size=vocab_size,
            diacritic_size=diacritic_size,
            max_sequence_length=model_hyperparams.get('max_sequence_length', MAX_SEQUENCE_LENGTH),
            embedding_dim=model_hyperparams.get('embedding_dim', EMBEDDING_DIM),
            d_model=model_hyperparams.get('d_model', EMBEDDING_DIM), # default d_model to embedding_dim
            nhead=model_hyperparams.get('nhead', NHEAD),
            num_encoder_layers=model_hyperparams.get('num_encoder_layers', NUM_ENCODER_LAYERS),
            dim_feedforward=model_hyperparams.get('dim_feedforward', DIM_FEEDFORWARD),
            transformer_dropout=model_hyperparams.get('transformer_dropout', 0.1),
            dropout_rate=model_hyperparams.get('dropout_rate', DROPOUT_RATE)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def diacritize_text(self, text):
        self.model.eval() # Ensure model is in eval mode
        input_text_clean = self.processor.strip_diacritics(text.strip())
        if not input_text_clean:
            return text # Return original if stripping results in empty

        encoded_input = self.processor.encode_input(input_text_clean, pad=True) # Pad for model
        input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(self.device)
        
        # Create padding mask for inference
        src_padding_mask = (input_tensor == self.processor.char_to_idx['<PAD>']).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, src_padding_mask=src_padding_mask)
            _, pred_indices = torch.max(outputs, dim=2)
        
        # Decode, ensuring to use the length of the *original cleaned input text*
        actual_len = len(input_text_clean)
        pred_diacritics_list = self.processor.decode_diacritics(pred_indices[0].cpu().numpy()[:actual_len])
        
        diacritized_text = self.processor.apply_diacritics(input_text_clean, pred_diacritics_list)
        return diacritized_text
    
    def batch_diacritize(self, texts):
        return [self.diacritize_text(text) for text in texts]


def load_csv_data_from_directory(directory_path, column_name='text_with_harakat'):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files found in directory: {directory_path}")
        return []
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    all_samples = []
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip') # skip bad lines
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {csv_file}. Skipping file.")
                continue
            
            # Drop NA and ensure text is string
            samples = df[column_name].dropna().astype(str).tolist()
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    print(f"Successfully loaded {len(all_samples)} text samples.")
    return all_samples


def train_model(
    # Data and Model Architecture Hyperparameters
    max_sequence_length=256, 
    embedding_dim=256, 
    # CNN parameters (if use_cnn_feature_extractor is True in model)
    # cnn_out_channels=256, # Example, align with d_model or make configurable
    # cnn_kernel_sizes=[3,5,7], # Example
    # Transformer parameters
    d_model=256, # Must match embedding_dim if no CNN or CNN output proj, or output of CNN
    nhead=8, 
    num_encoder_layers=6,
    dim_feedforward=2048, # Typical is 4*d_model
    transformer_dropout=0.1,
    # Training Hyperparameters
    batch_size=32, 
    dropout_rate=0.1, # General dropout for embeddings/CNNs if used
    learning_rate=5e-5, # Often smaller LR for transformers
    epochs=50, # Reduced default for faster testing; use more for real training
    patience=10, # Increased patience
    # Paths and Config
    data_directory="data/",
    results_dir="results/",
    model_type="AdvancedArabicDiacritizationModel" # To specify which model to use
):
    # Ensure d_model is consistent with embedding_dim for the new model
    if d_model != embedding_dim and model_type == "AdvancedArabicDiacritizationModel":
        print(f"Warning: For AdvancedArabicDiacritizationModel, d_model ({d_model}) is often set to embedding_dim ({embedding_dim}) if CNNs are simple or bypassed. Ensure consistency.")
        # If your CNNs project to a different dimension, d_model should match that.
        # For this version, our Advanced Model's CNN output `d_model` channels.
        # And its Positional Encoding is also `d_model`. Embedding is `embedding_dim`.
        # The `conv_extractor` maps `embedding_dim` to `d_model`.
        pass


    model_name_suffix = f"L{max_sequence_length}_B{batch_size}_E{embedding_dim}_D{d_model}_H{nhead}_TEnc{num_encoder_layers}_FF{dim_feedforward}_Drop{dropout_rate}_LR{learning_rate}"
    model_name = f"AdvArabicDiacritizer_{model_name_suffix}"
    print(f"\n{'='*80}\nTraining model: {model_name}\n{'='*80}")
    
    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_path = f"checkpoints/{model_name}/{model_name}.pt"
    processor_path = f"models/{model_name}/{model_name}_processor.pkl"
    final_model_path = f"models/{model_name}/{model_name}_final.pt" # Changed from model_path to avoid conflict

    print("1. Loading data...")
    diacritized_texts = load_csv_data_from_directory(data_directory)
    if not diacritized_texts:
        print("No data loaded. Exiting training.")
        return None
    # Optional: Filter very short texts or apply other cleaning
    diacritized_texts = [text for text in diacritized_texts if len(text) > 5] # Basic filter


    print("2. Initializing data processor...")
    data_processor = ArabicDiacriticsDataProcessor(max_sequence_length=max_sequence_length)
    
    print("3. Preparing dataset...")
    inputs, targets = data_processor.prepare_dataset(diacritized_texts)
    if not inputs:
        print("Dataset preparation resulted in no usable data. Exiting.")
        return None

    # Save processor once vocabulary is built
    # This should be done *after* prepare_dataset which calls create_character_mappings
    with open(processor_path, 'wb') as f:
        pickle.dump(data_processor, f)
    print(f"Data processor saved to {processor_path}")
    
    print("4. Applying data augmentation (if configured)...") # Augmentation factor is inside processor
    augmented_inputs, augmented_targets = data_processor.data_augmentation(inputs, targets, augmentation_factor=0.05) # Small augmentation
    
    print("5. Splitting dataset...")
    # Test set first
    inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
        augmented_inputs, augmented_targets, test_size=0.10, random_state=42 # Smaller test set
    )
    # Train and Validation from the rest
    val_split_ratio = 0.10 / (1-0.10) # e.g. 10% of original as val -> 0.1 / 0.9 of (train+val)
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs_train_val, targets_train_val, test_size=val_split_ratio, random_state=42
    )
    
    print(f"Train set: {len(inputs_train)} samples")
    print(f"Validation set: {len(inputs_val)} samples")
    print(f"Test set: {len(inputs_test)} samples")

    if not inputs_train or not inputs_val or not inputs_test:
        print("One of the data splits is empty. Check data size and split ratios. Exiting.")
        return None

    pad_idx_char = data_processor.char_to_idx['<PAD>']
    # Target padding: use the index of "no diacritic" for padding targets for loss ignore_index.
    # The DiacritizationTrainer's criterion ignore_index should match this.
    # User's original code padded targets with diacritic_to_idx[""]. Let's find its index.
    pad_idx_target = data_processor.diacritic_to_idx.get("", 0) # Default to 0 if "" somehow not in map

    print("6. Creating PyTorch datasets and dataloaders...")
    train_dataset = ArabicDiacriticsDataset(inputs_train, targets_train, data_processor)
    val_dataset = ArabicDiacriticsDataset(inputs_val, targets_val, data_processor)
    test_dataset = ArabicDiacriticsDataset(inputs_test, targets_test, data_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2) # No shuffle for test

    print("7. Initializing model...")
    if model_type == "AdvancedArabicDiacritizationModel":
        model = AdvancedArabicDiacritizationModel(
            vocab_size=data_processor.get_vocab_size(),
            diacritic_size=data_processor.get_diacritic_size(),
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            transformer_dropout=transformer_dropout,
            dropout_rate=dropout_rate
        )
    else: # Fallback to user's original model structure if specified or by default
        # This part would need the old SimplifiedArabicDiacritizationModel definition
        print(f"Warning: Model type '{model_type}' not fully configured for Advanced. Ensure definition exists.")
        # For the purpose of this upgrade, we assume Advanced model is used.
        # If you want to use Simplified, ensure its class definition is available.
        # model = SimplifiedArabicDiacritizationModel(vocab_size=..., diacritic_size=..., ...)
        raise ValueError(f"Unsupported model_type: {model_type}. This script focuses on AdvancedArabicDiacritizationModel.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type}, Total trainable parameters: {total_params:,}")
    
    print("8. Training model...")
    start_time = time.time()
    trainer = DiacritizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        pad_idx=pad_idx_target # Pass target pad index for loss ignore
    )
    
    model, history = trainer.train( # Trainer now returns history
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        patience=patience
    )
    training_time = time.time() - start_time
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if history.get('val_loss'): plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    if history.get('val_accuracy'): plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Curve')
    plt.tight_layout()
    history_plot_path = f"{results_dir}/{model_name}_training_history.png"
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    plt.close()


    print("9. Evaluating model on test set...")
    evaluator = DiacritizationEvaluator(model, data_processor, device, pad_idx=pad_idx_target)
    inference_start_time = time.time()
    # Pass original unencoded test inputs/targets for word-level metrics and examples
    metrics = evaluator.evaluate(test_loader, inputs_test, targets_test)
    inference_time = time.time() - inference_start_time
    
    print("10. Saving final model and processor (if not already saved)...")
    # Processor is already saved after prepare_dataset
    # Save final model (could be the best one from checkpoint, or last epoch if no val)
    # The trainer already loads the best model.
    
    # Save hyperparams with the final model for easier loading
    hyperparameters_saved = {
        'max_sequence_length': max_sequence_length,
        'embedding_dim': embedding_dim,
        'd_model': d_model,
        'nhead': nhead,
        'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'transformer_dropout': transformer_dropout,
        'dropout_rate': dropout_rate, # General dropout
        'vocab_size': data_processor.get_vocab_size(),
        'diacritic_size': data_processor.get_diacritic_size()
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters_saved, # Save key hyperparams
        'epoch': trainer.model.epoch if hasattr(trainer.model, 'epoch') else epochs # If loaded from checkpoint
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    results = {
        'model_name': model_name,
        'hyperparameters_used_for_run': {
            'max_sequence_length': max_sequence_length, 'batch_size': batch_size,
            'embedding_dim': embedding_dim, 'd_model': d_model, 'nhead': nhead,
            'num_encoder_layers': num_encoder_layers, 'dim_feedforward': dim_feedforward,
            'transformer_dropout': transformer_dropout, 'dropout_rate': dropout_rate,
            'learning_rate': learning_rate, 'epochs_run': epochs, 'patience': patience
        },
        'metrics': metrics,
        'model_params_count': total_params,
        'training_time_seconds': training_time,
        'avg_inference_time_per_batch_seconds': inference_time / len(test_loader) if test_loader else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_json_path = f"{results_dir}/{model_name}_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Training complete for {model_name}!")
    print(f"Results saved to {results_json_path}")
    
    return results


def run_hyperparameter_optimization():
    """Run hyperparameter optimization with multiple combinations"""
    
    # Define hyperparameter grid - focus on Transformer parts
    param_grid = {
        'max_sequence_length': [128, 256], # Shorter sequences are faster to train
        'batch_size': [32, 64],
        'embedding_dim': [256], # Keep fixed for now, d_model will vary or match
        'd_model': [256], # Tied to embedding_dim for this setup or output of CNN
        'nhead': [4, 8],
        'num_encoder_layers': [3, 6], # Fewer layers for faster iteration
        'dim_feedforward': [1024, 2048], # Usually 2x to 4x d_model
        'dropout_rate': [0.1, 0.2], # General dropout
        'transformer_dropout': [0.1, 0.2], # Transformer specific dropout
        'learning_rate': [1e-4, 5e-5],
        'epochs': [30],  # Reduced epochs for HPO runs
        'patience': [5]  # Shorter patience for HPO
    }
    
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_hpo = f"results/hpo_advanced_{timestamp}"
    os.makedirs(results_dir_hpo, exist_ok=True)
    
    with open(f"{results_dir_hpo}/param_grid.json", 'w', encoding='utf-8') as f:
        json.dump(param_grid, f, indent=4, ensure_ascii=False)
    
    all_run_results = []
    print(f"Starting hyperparameter optimization with {len(combinations)} combinations for Advanced Model.")
    
    for i, combo in enumerate(combinations):
        print(f"\nCombination {i+1}/{len(combinations)}")
        params = dict(zip(keys, combo))
        
        # Ensure d_model is consistent if tied to embedding_dim, or set based on CNN output design
        # In our current AdvancedArabicDiacritizationModel, embedding_dim feeds CNN,
        # and CNN output (and thus d_model for Transformer) is `params['d_model']`.
        # So `embedding_dim` and `d_model` from grid are used directly.

        try:
            run_result = train_model(
                **params, 
                results_dir=results_dir_hpo, 
                model_type="AdvancedArabicDiacritizationModel",
                data_directory="data/" # Make sure this path is correct
            )
            if run_result:
                all_run_results.append(run_result)
        except Exception as e:
            print(f"Error training model with parameters {params}: {e}")
            import traceback
            traceback.print_exc()
            with open(f"{results_dir_hpo}/errors.log", 'a', encoding='utf-8') as f:
                f.write(f"Error with parameters {params}: {str(e)}\n{traceback.format_exc()}\n")
    
    if all_run_results:
        with open(f"{results_dir_hpo}/all_hpo_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_run_results, f, indent=4, ensure_ascii=False)
        create_comparison_report(all_run_results, results_dir_hpo)
    else:
        print("No results from HPO runs.")
        
    return all_run_results


def create_comparison_report(results_list, report_dir):
    if not results_list:
        print("No results to compare.")
        return None
    
    comparison_data = []
    for res in results_list:
        if res and 'metrics' in res and 'hyperparameters_used_for_run' in res : # check if res is not None
             # Check if metrics dict exists and has the key
            char_acc = res['metrics'].get('character_accuracy', 0.0)
            word_acc = res['metrics'].get('word_accuracy', 0.0)
            der = res['metrics'].get('diacritic_error_rate', 1.0)


            entry = {
                'model_name': res.get('model_name', 'N/A'),
                'character_accuracy': char_acc,
                'word_accuracy': word_acc,
                'diacritic_error_rate': der,
                'model_params_count': res.get('model_params_count', 0),
                'training_time_seconds': res.get('training_time_seconds', 0),
                **res.get('hyperparameters_used_for_run', {}) # Flatten hyperparams
            }
            comparison_data.append(entry)
        else:
            print(f"Skipping malformed result: {res}")

    if not comparison_data:
        print("No valid result data for comparison report.")
        return None

    df = pd.DataFrame(comparison_data)
    df_sorted = df.sort_values('character_accuracy', ascending=False)
    
    df_sorted.to_csv(f"{report_dir}/hpo_comparison.csv", index=False, encoding='utf-8')
    print(f"HPO comparison report saved to {report_dir}/hpo_comparison.csv")
    
    create_visualizations(df_sorted, report_dir) # Use the sorted dataframe
    
    print("\nTop 3 models from HPO by character accuracy:")
    for idx, row in df_sorted.head(3).iterrows():
        print(f"{idx+1}. {row['model_name']} - Char Acc: {row['character_accuracy']:.4f}, Word Acc: {row['word_accuracy']:.4f}, DER: {row['diacritic_error_rate']:.4f}")
    
    return df_sorted

def create_visualizations(df, viz_results_dir):
    if df.empty:
        print("DataFrame for visualizations is empty.")
        return

    viz_dir = f"{viz_results_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    plt.style.use('ggplot')

    # 1. Accuracy vs Model Size
    if 'model_params_count' in df.columns and 'character_accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='model_params_count', y='character_accuracy', hue='num_encoder_layers', size='nhead', alpha=0.7, legend="brief")
        plt.xlabel('Model Size (parameters)')
        plt.ylabel('Character Accuracy')
        plt.title('Model Size vs. Accuracy (Colored by Encoder Layers, Sized by Heads)')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/model_size_vs_accuracy.png", dpi=300)
        plt.close()

    # 2. Training Time vs Accuracy
    if 'training_time_seconds' in df.columns and 'character_accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=df['training_time_seconds']/60, y='character_accuracy', hue='learning_rate', style='batch_size', alpha=0.7, legend="brief")
        plt.xlabel('Training Time (minutes)')
        plt.ylabel('Character Accuracy')
        plt.title('Training Time vs. Accuracy (Colored by LR, Styled by Batch Size)')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/training_time_vs_accuracy.png", dpi=300)
        plt.close()

    # 3. Hyperparameter Impact Plots (Box plots or Violin plots)
    key_hyperparams = ['nhead', 'num_encoder_layers', 'dim_feedforward', 'learning_rate', 'dropout_rate', 'transformer_dropout']
    for param in key_hyperparams:
        if param in df.columns and 'character_accuracy' in df.columns:
            if df[param].nunique() > 1: # Only plot if there's variation
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x=param, y='character_accuracy')
                plt.title(f'Impact of {param} on Character Accuracy')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/{param}_impact_on_accuracy.png", dpi=300)
                plt.close()
    
    # 4. Bar chart of top N models
    top_n = min(10, len(df))
    if top_n > 0 and 'model_name' in df.columns and 'character_accuracy' in df.columns:
        top_models_df = df.nlargest(top_n, 'character_accuracy')
        plt.figure(figsize=(14, 7))
        bars = plt.bar(top_models_df['model_name'], top_models_df['character_accuracy'], color=sns.color_palette("viridis", top_n))
        plt.xlabel('Model Configuration')
        plt.ylabel('Character Accuracy')
        plt.title(f'Top {top_n} Models by Character Accuracy')
        plt.xticks(rotation=75, ha='right', fontsize=8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
        plt.ylim(top=max(0.6, df['character_accuracy'].max() * 1.05) if not df['character_accuracy'].empty else 1.0) # Adjust y-limit
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/top_{top_n}_models_accuracy.png", dpi=300)
        plt.close()

    # 5. Correlation matrix (numeric columns only)
    numeric_df = df.select_dtypes(include=np.number)
    # Include only relevant metrics and hyperparameters for correlation
    cols_for_corr = [
        'character_accuracy', 'word_accuracy', 'diacritic_error_rate', 
        'model_params_count', 'training_time_seconds',
        'max_sequence_length', 'batch_size', 'embedding_dim', 'd_model', 
        'nhead', 'num_encoder_layers', 'dim_feedforward', 
        'dropout_rate', 'transformer_dropout', 'learning_rate'
    ]
    # Filter numeric_df to include only existing columns from cols_for_corr
    valid_cols_for_corr = [col for col in cols_for_corr if col in numeric_df.columns]

    if len(valid_cols_for_corr) > 1:
        corr_matrix = numeric_df[valid_cols_for_corr].corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
        plt.title('Correlation Matrix of Hyperparameters and Metrics')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/correlation_matrix.png", dpi=300)
        plt.close()
    print(f"Visualizations saved to {viz_dir}")


def main():
    print("Advanced Arabic Diacritization Neural Network - Training and HPO")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Create a dummy data directory and a CSV file if it doesn't exist for testing
    # In a real scenario, the user provides this data.
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not glob.glob("data/*.csv"):
        print("No CSV data found in 'data/' directory. Creating a small dummy dataset for demonstration.")
        dummy_data = {
            'text_with_harakat': [
                "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ.",
                "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ.",
                "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ.",
                "اللَّهُ أَكْبَرُ.",
                "لاَ إِلَهَ إِلاَّ اللَّهُ.",
                "مَرْحَبًا بِكُمْ فِي الْعَالَمِ الْعَرَبِيِّ.",
                "كَيْفَ حَالُكَ الْيَوْمَ؟",
                "أَنَا بِخَيْرٍ، شُكْرًا لَكَ.",
                "مَا اسْمُكَ؟",
                "اسْمِي أَحْمَدُ."
            ] * 20 # Multiply to have a bit more data for splits
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv("data/dummy_dataset.csv", index=False, encoding='utf-8')
        print("Created data/dummy_dataset.csv for demonstration.")
        print("Please replace this with your actual dataset for meaningful results.")


    # Option 1: Run a single training with specific (or default) parameters
    print("\n--- Running Single Training Run with Default Advanced Settings ---")
    train_model(
        epochs=500, # Short epochs for quick test
        data_directory="data/",
        results_dir="results/single_run_advanced",
        model_type="AdvancedArabicDiacritizationModel",
        # You can override other defaults here, e.g.,
        # batch_size=16, learning_rate=1e-4
    )

    # Option 2: Run Hyperparameter Optimization
    # print("\n--- Running Hyperparameter Optimization for Advanced Model ---")
    # results = run_hyperparameter_optimization()
    
    # if results:
    #     print("\nHyperparameter optimization complete!")
    #     # The HPO function already prints location of results.
    # else:
    #     print("\nHyperparameter optimization did not produce any results or was not run.")

    # Example of how to use the Diacritizer for inference after training
    # Find the best model from HPO or a specific trained model
    # This is just a placeholder for how you might load and use it.
    # You'd typically find the best model's path from the HPO results CSV.
    print("\n--- Example Inference (Placeholder) ---")
    # Replace with actual paths from your training/HPO
    example_model_path = "models/AdvArabicDiacritizer_..._final.pt" # Path to a trained .pt model
    example_processor_path = "models/AdvArabicDiacritizer_..._processor.pkl" # Path to its .pkl processor
    
    if os.path.exists(example_model_path) and os.path.exists(example_processor_path):
        try:
            print(f"Loading Diacritizer with model: {example_model_path}")
            diacritizer = ArabicDiacritizer(model_path=example_model_path, processor_path=example_processor_path)
            sample_text = "السلام عليكم"
            diacritized_sample = diacritizer.diacritize_text(sample_text)
            print(f"Original: {sample_text}")
            print(f"Diacritized: {diacritized_sample}")
        except FileNotFoundError:
            print(f"Could not find example model/processor for inference demo. Please train a model first.")
            print(f"Look for paths like: {example_model_path} and {example_processor_path}")
        except Exception as e:
            print(f"Error during example inference: {e}")
    else:
        print("Example model/processor paths not found. Run training/HPO to generate them.")
        print("After HPO, check 'results/hpo_advanced_TIMESTAMP/hpo_comparison.csv' for best model names,")
        print("then find corresponding files in 'models/MODEL_NAME/'.")


if __name__ == "__main__":
    main()