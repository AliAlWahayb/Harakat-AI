import torch
import argparse
import os
import time
from pytorch_arabic_diacritization import ArabicDiacriticsDataProcessor, SimplifiedArabicDiacritizationModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(model_path, processor_path):
    """Load the trained model and processor"""
    print(f"Loading model from {model_path}...")
    
    # Load processor
    import pickle
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = SimplifiedArabicDiacritizationModel(
        vocab_size=len(processor.char_to_idx),
        diacritic_size=len(processor.diacritic_to_idx)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, processor

def diacritize_text(text, model, processor):
    """Add diacritics to input text"""
    # Preprocess text
    input_text = text.strip()
    
    # Encode input
    encoded_input = processor.encode_input(input_text)
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)
    
    # Get predictions
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_indices = torch.max(outputs, dim=2)
    inference_time = time.time() - start_time
    
    # Convert indices to diacritics
    pred_diacritics = processor.decode_diacritics(pred_indices[0].cpu().numpy()[:len(input_text)])
    
    # Apply diacritics to input text
    diacritized_text = processor.apply_diacritics(input_text, pred_diacritics)
    
    print(f"Inference time: {inference_time * 1000:.2f} ms")
    return diacritized_text

def main():
    parser = argparse.ArgumentParser(description='Arabic Diacritization Inference')
    parser.add_argument('--model_path', type=str, default='models/pytorch_diacritization.pt', help='Path to the trained model')
    parser.add_argument('--processor_path', type=str, default='models/pytorch_diacritization_processor.pkl', help='Path to the data processor')
    parser.add_argument('--text', type=str, default=None, help='Arabic text to add diacritics to')
    parser.add_argument('--input_file', type=str, default=None, help='Input file containing Arabic text')
    parser.add_argument('--output_file', type=str, default=None, help='Output file to save diacritized text')
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor = load_model(args.model_path, args.processor_path)
    
    # Process text
    if args.text:
        # Process single text
        result = diacritize_text(args.text, model, processor)
        print(f"Original: {args.text}")
        print(f"Diacritized: {result}")
    
    elif args.input_file:
        # Process file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = []
        for line in lines:
            line = line.strip()
            if line:
                result = diacritize_text(line, model, processor)
                results.append(result)
                print(f"Original: {line}")
                print(f"Diacritized: {result}")
                print()
        
        # Save to output file if specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result + '\n')
            print(f"Results saved to {args.output_file}")
    
    else:
        # Interactive mode
        print("Enter Arabic text to add diacritics (type 'exit' to quit):")
        while True:
            text = input("> ")
            if text.lower() == 'exit':
                break
            if text:
                result = diacritize_text(text, model, processor)
                print(f"Diacritized: {result}")
                print()

if __name__ == "__main__":
    main()