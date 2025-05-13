import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm

class ArabicDiacritizer:
    def __init__(self, model_path, processor_path):
        """
        Initialize the Arabic diacritizer with a trained model and processor.
        
        Args:
            model_path (str): Path to the trained model file
            processor_path (str): Path to the saved data processor
        """
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        print(f"Loading processor from {processor_path}...")
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        print("Initialization complete.")
    
    def diacritize_text(self, text, batch_size=1):
        """
        Add diacritics to input text.
        
        Args:
            text (str): Input text without diacritics
            batch_size (int): Batch size for processing long texts
        
        Returns:
            str: Text with diacritics
        """
        # Preprocess text
        input_text = text.strip()
        
        # For very long texts, process in chunks to avoid memory issues
        if len(input_text) > self.processor.max_sequence_length:
            return self._process_long_text(input_text, batch_size)
        
        # Encode input
        encoded_input = np.array([self.processor.encode_input(input_text)])
        
        # Get predictions
        predictions = self.model.predict(encoded_input, verbose=0)
        pred_indices = np.argmax(predictions[0], axis=-1)
        
        # Convert indices to diacritics
        pred_diacritics = self.processor.decode_diacritics(pred_indices[:len(input_text)])
        
        # Apply diacritics to input text
        diacritized_text = self.processor.apply_diacritics(input_text, pred_diacritics)
        
        return diacritized_text
    
    def _process_long_text(self, text, batch_size=1):
        """
        Process long text by splitting it into sentences or chunks.
        
        Args:
            text (str): Long input text
            batch_size (int): Number of chunks to process at once
        
        Returns:
            str: Diacritized text
        """
        # Try to split by sentences
        sentences = text.replace('؟', '؟\n').replace('!', '!\n').replace('.', '.\n').split('\n')
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.processor.max_sequence_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If sentence is longer than max_sequence_length, split it further
                if len(sentence) > self.processor.max_sequence_length:
                    # Split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 < self.processor.max_sequence_length:
                            current_chunk += word + " "
                        else:
                            chunks.append(current_chunk)
                            current_chunk = word + " "
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process chunks in batches
        diacritized_chunks = []
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing text chunks"):
            batch_chunks = chunks[i:i+batch_size]
            
            # Encode inputs
            encoded_inputs = np.array([self.processor.encode_input(chunk) for chunk in batch_chunks])
            
            # Get predictions
            predictions = self.model.predict(encoded_inputs, verbose=0)
            
            # Process each prediction
            for j, chunk in enumerate(batch_chunks):
                pred_indices = np.argmax(predictions[j], axis=-1)
                pred_diacritics = self.processor.decode_diacritics(pred_indices[:len(chunk)])
                diacritized_chunk = self.processor.apply_diacritics(chunk, pred_diacritics)
                diacritized_chunks.append(diacritized_chunk)
        
        # Join all diacritized chunks
        return ''.join(diacritized_chunks)
    
    def diacritize_file(self, input_file, output_file=None, batch_size=1):
        """
        Diacritize text from a file and save the result.
        
        Args:
            input_file (str): Path to input file
            output_file (str, optional): Path to output file. If None, will use input_file + '.diacritized'
            batch_size (int): Batch size for processing
        
        Returns:
            str: Path to the output file
        """
        # Set default output file if not provided
        if output_file is None:
            output_file = input_file + '.diacritized'
        
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Diacritize text
        diacritized_text = self.diacritize_text(text, batch_size)
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(diacritized_text)
        
        return output_file
    
    def batch_diacritize(self, texts, batch_size=32):
        """
        Diacritize a batch of texts.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
        
        Returns:
            list: List of diacritized texts
        """
        results = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batch"):
            batch_texts = texts[i:i+batch_size]
            
            # Handle texts that are too long
            processed_batch = []
            for text in batch_texts:
                if len(text) > self.processor.max_sequence_length:
                    processed_batch.append(self._process_long_text(text))
                else:
                    processed_batch.append(text)
            
            # Encode inputs
            encoded_inputs = np.array([
                self.processor.encode_input(text) for text in processed_batch
            ])
            
            # Get predictions
            predictions = self.model.predict(encoded_inputs, verbose=0)
            
            # Process each prediction
            for j, text in enumerate(processed_batch):
                pred_indices = np.argmax(predictions[j], axis=-1)
                pred_diacritics = self.processor.decode_diacritics(pred_indices[:len(text)])
                diacritized_text = self.processor.apply_diacritics(text, pred_diacritics)
                results.append(diacritized_text)
        
        return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Arabic Text Diacritization')
    parser.add_argument('--model', type=str, default='models/arabic_diacritization_simplified.keras',
                        help='Path to the trained model')
    parser.add_argument('--processor', type=str, default='models/arabic_diacritization_processor.pkl',
                        help='Path to the saved data processor')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file or directory (default: input + .diacritized)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing (default: 1)')
    
    args = parser.parse_args()
    
    # Check if model and processor exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return 1
    
    if not os.path.exists(args.processor):
        print(f"Error: Processor file not found at {args.processor}")
        return 1
    
    # Initialize diacritizer
    diacritizer = ArabicDiacritizer(args.model, args.processor)
    
    # Process input
    if os.path.isfile(args.input):
        # Process single file
        output_file = args.output or args.input + '.diacritized'
        print(f"Diacritizing file: {args.input} -> {output_file}")
        diacritizer.diacritize_file(args.input, output_file, args.batch_size)
        print(f"Diacritization complete. Output saved to {output_file}")
    
    elif os.path.isdir(args.input):
        # Process directory
        output_dir = args.output or args.input + '_diacritized'
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all text files in the directory
        text_files = [f for f in os.listdir(args.input) if f.endswith('.txt')]
        print(f"Found {len(text_files)} text files in {args.input}")
        
        for file in text_files:
            input_file = os.path.join(args.input, file)
            output_file = os.path.join(output_dir, file)
            print(f"Diacritizing file: {input_file} -> {output_file}")
            diacritizer.diacritize_file(input_file, output_file, args.batch_size)
        
        print(f"Diacritization complete. Output saved to {output_dir}")
    
    else:
        print(f"Error: Input path not found: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

    ## Process a file: python inference.py --input sample.txt --model models/arabic_diacritization_simplified.keras --processor models/arabic_diacritization_processor.pkl