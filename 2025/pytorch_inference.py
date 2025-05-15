import torch
import os
import time
import pickle
import arabic_reshaper
from bidi.algorithm import get_display
from colorama import Fore, Style, init
import unicodedata

# Import the required classes from your module
# Make sure this path is correct for your project structure
from pytorch_arabic_diacritization import ArabicDiacriticsDataProcessor, SimplifiedArabicDiacritizationModel

# Initialize colorama for colored terminal output
init()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{Fore.CYAN}Using device: {device}{Style.RESET_ALL}")

def load_model(model_path, processor_path):
    """Load the trained model and processor"""
    print(f"{Fore.YELLOW}Loading model from {model_path}...{Style.RESET_ALL}")
    
    # Load processor
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
    
    print(f"{Fore.GREEN}Inference time: {inference_time * 1000:.2f} ms{Style.RESET_ALL}")
    return diacritized_text, pred_diacritics

def display_arabic(text):
    """Display Arabic text properly in the terminal with RTL support"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def print_header():
    """Print a stylish header for the application"""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{' ' * 15}ARABIC DIACRITIZATION TOOL")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

def list_available_models():
    """List all available models in the models directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    available_models = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(models_dir, model_dir, f"{model_dir}.pt")
        processor_path = os.path.join(models_dir, model_dir, f"{model_dir}_processor.pkl")
        
        if os.path.exists(model_path) and os.path.exists(processor_path):
            available_models.append((model_dir, model_path, processor_path))
    
    return available_models

def show_unicode_details(text):
    """Show detailed Unicode information for each character"""
    details = []
    for i, char in enumerate(text):
        char_name = unicodedata.name(char, "UNKNOWN")
        char_code = f"U+{ord(char):04X}"
        details.append(f"{i+1}: {char} - {char_code} - {char_name}")
    return details

def main():
    print_header()
    
    # List available models
    available_models = list_available_models()
    
    if not available_models:
        print(f"{Fore.RED}No models found in the 'models' directory.{Style.RESET_ALL}")
        print("Please make sure you have trained models available.")
        return
    
    # Display available models
    print(f"{Fore.YELLOW}Available models:{Style.RESET_ALL}")
    for i, (model_name, _, _) in enumerate(available_models):
        print(f"{i+1}. {model_name}")
    
    # Let user select a model
    while True:
        try:
            selection = input(f"\n{Fore.CYAN}Select a model (1-{len(available_models)}), or press Enter for default (1): {Style.RESET_ALL}")
            if selection == "":
                selection = 1
            else:
                selection = int(selection)
            
            if 1 <= selection <= len(available_models):
                model_name, model_path, processor_path = available_models[selection-1]
                break
            else:
                print(f"{Fore.RED}Invalid selection. Please try again.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
    
    # Load the selected model
    model, processor = load_model(model_path, processor_path)
    print(f"{Fore.GREEN}Model '{model_name}' loaded successfully!{Style.RESET_ALL}")
    
    # Interactive mode
    print(f"\n{Fore.CYAN}Enter Arabic text to add diacritics:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}(Type 'exit' to quit, 'file' to process a file, 'help' for commands){Style.RESET_ALL}")
    
    while True:
        print(f"\n{Fore.YELLOW}> {Style.RESET_ALL}", end="")
        text = input()
        
        if text.lower() == 'exit':
            break
        elif text.lower() == 'help':
            print(f"\n{Fore.CYAN}Available commands:{Style.RESET_ALL}")
            print("  exit - Exit the program")
            print("  file - Process a text file")
            print("  model - Change the current model")
            print("  help - Show this help message")
            print("  debug - Toggle detailed Unicode debugging")
            continue
        elif text.lower() == 'model':
            # Let user select a different model
            print(f"\n{Fore.YELLOW}Available models:{Style.RESET_ALL}")
            for i, (model_name, _, _) in enumerate(available_models):
                print(f"{i+1}. {model_name}")
            
            try:
                selection = input(f"\n{Fore.CYAN}Select a model (1-{len(available_models)}): {Style.RESET_ALL}")
                selection = int(selection)
                
                if 1 <= selection <= len(available_models):
                    model_name, model_path, processor_path = available_models[selection-1]
                    model, processor = load_model(model_path, processor_path)
                    print(f"{Fore.GREEN}Model '{model_name}' loaded successfully!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid selection.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            continue
        elif text.lower() == 'file':
            # Process a file
            file_path = input(f"{Fore.CYAN}Enter the path to the input file: {Style.RESET_ALL}")
            if not os.path.exists(file_path):
                print(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
                continue
            
            output_path = input(f"{Fore.CYAN}Enter the path for the output file (or press Enter to print to console): {Style.RESET_ALL}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                results = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        print(f"\n{Fore.YELLOW}Processing line {i+1}/{len(lines)}{Style.RESET_ALL}")
                        result, _ = diacritize_text(line, model, processor)
                        results.append(result)
                        
                        # Display original and diacritized text
                        print(f"{Fore.CYAN}Original: {Style.RESET_ALL}{display_arabic(line)}")
                        print(f"{Fore.GREEN}Diacritized: {Style.RESET_ALL}{display_arabic(result)}")
                
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for result in results:
                            f.write(result + '\n')
                    print(f"\n{Fore.GREEN}Results saved to {output_path}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error processing file: {str(e)}{Style.RESET_ALL}")
            continue
        
        if not text:
            continue
        
        # Process the text
        result, diacritics = diacritize_text(text, model, processor)
        
        # Display original and diacritized text with proper RTL formatting
        print(f"\n{Fore.CYAN}Original:{Style.RESET_ALL}")
        print(display_arabic(text))
        
        print(f"\n{Fore.GREEN}Diacritized:{Style.RESET_ALL}")
        print(display_arabic(result))
        
        # Show the difference more clearly
        print(f"\n{Fore.YELLOW}Verification (character by character):{Style.RESET_ALL}")
        for i, (orig_char, diac_char) in enumerate(zip(text, result)):
            if orig_char != diac_char:
                print(f"Position {i+1}: '{orig_char}' → '{diac_char}'")
        
        # Show diacritics added
        print(f"\n{Fore.YELLOW}Diacritics added:{Style.RESET_ALL}")
        diac_count = sum(1 for d in diacritics if d != '')
        print(f"Added {diac_count} diacritics to {len(text)} characters")
        
        # Show Unicode details for better debugging
        print(f"\n{Fore.YELLOW}Unicode details of diacritized text:{Style.RESET_ALL}")
        for detail in show_unicode_details(result):
            print(detail)
        
        # Save to file option
        save_option = input(f"\n{Fore.CYAN}Save this result to a file? (y/n): {Style.RESET_ALL}")
        if save_option.lower() == 'y':
            file_name = input(f"{Fore.CYAN}Enter file name: {Style.RESET_ALL}")
            if not file_name:
                file_name = "diacritized_output.txt"
            
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"{Fore.GREEN}Result saved to {file_name}{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program terminated by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n{Fore.CYAN}Thank you for using the Arabic Diacritization Tool!{Style.RESET_ALL}")