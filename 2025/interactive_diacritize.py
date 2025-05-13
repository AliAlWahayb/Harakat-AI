import os
import tensorflow as tf
import numpy as np
import pickle
import argparse

def load_diacritizer(model_path, processor_path):
    """Load the diacritizer model and processor"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading processor from {processor_path}...")
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    return model, processor

def diacritize_text(text, model, processor):
    """Add diacritics to input text"""
    # Preprocess text
    input_text = text.strip()
    
    # Encode input
    encoded_input = np.array([processor.encode_input(input_text)])
    
    # Get predictions
    predictions = model.predict(encoded_input, verbose=0)
    pred_indices = np.argmax(predictions[0], axis=-1)
    
    # Convert indices to diacritics
    pred_diacritics = processor.decode_diacritics(pred_indices[:len(input_text)])
    
    # Apply diacritics to input text
    diacritized_text = processor.apply_diacritics(input_text, pred_diacritics)
    
    return diacritized_text

def main():
    parser = argparse.ArgumentParser(description='Interactive Arabic Text Diacritization')
    parser.add_argument('--model', type=str, default='models/arabic_diacritization_simplified.keras',
                        help='Path to the trained model')
    parser.add_argument('--processor', type=str, default='models/arabic_diacritization_processor.pkl',
                        help='Path to the saved data processor')
    
    args = parser.parse_args()
    
    # Check if model and processor exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return 1
    
    if not os.path.exists(args.processor):
        print(f"Error: Processor file not found at {args.processor}")
        return 1
    
    # Load model and processor
    model, processor = load_diacritizer(args.model, args.processor)
    
    print("\n=== Arabic Text Diacritization ===")
    print("Type 'exit' or 'quit' to end the program.")
    print("Enter Arabic text to add diacritics:")
    
    while True:
        # Get input from user
        text = input("\n> ")
        
        # Check if user wants to exit
        if text.lower() in ['exit', 'quit']:
            break
        
        # Skip empty input
        if not text.strip():
            continue
        
        try:
            # Diacritize text
            diacritized = diacritize_text(text, model, processor)
            
            # Print result
            print("\nOriginal:")
            print(text)
            print("\nDiacritized:")
            print(diacritized)
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("Goodbye!")
    return 0

if __name__ == "__main__":
    main()

    """
    Interactive mode
    python interactive_diacritize.py --model models/arabic_diacritization_simplified.keras --processor models/arabic_diacritization_processor.pkl
    """