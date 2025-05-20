import torch
import os
import time
import pickle
import gradio as gr
import arabic_reshaper
from bidi.algorithm import get_display
from pytorch_arabic_diacritization import ArabicDiacriticsDataProcessor, SimplifiedArabicDiacritizationModel
import re
import unicodedata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model and processor
current_model = None
current_processor = None

def remove_tatweel(text):
    """Remove Tatweel character explicitly"""
    return text.replace('ـ', '')

def remove_harakat_after_special_chars(text):
    """
    Remove Arabic diacritics (harakat) ONLY if they follow special characters.
    Special chars are non-word, non-Arabic letters, non-space characters.
    """
    # Include Arabic question mark (؟) and other punctuation in special characters
    special_chars_pattern = r'[^\w\s\u0600-\u06FF]|؟|،'
    normalized = unicodedata.normalize('NFD', text)
    
    result = []
    i = 0
    length = len(normalized)
    
    while i < length:
        char = normalized[i]
        # If current char is special char or Arabic punctuation
        if re.match(special_chars_pattern, char):
            result.append(char)
            i += 1
            # Skip any following harakat combining marks
            while i < length and unicodedata.category(normalized[i]) == 'Mn' and '\u064B' <= normalized[i] <= '\u0652':
                i += 1
        else:
            result.append(char)
            i += 1
    
    # Recompose back to NFC form
    return unicodedata.normalize('NFC', ''.join(result))

def load_model(model_path, processor_path):
    """Load the trained model and processor"""
    global current_model, current_processor
    
    with open(processor_path, 'rb') as f:
        current_processor = pickle.load(f)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SimplifiedArabicDiacritizationModel(
        vocab_size=len(current_processor.char_to_idx),
        diacritic_size=len(current_processor.diacritic_to_idx)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    current_model = model
    
    return "Model loaded successfully!"

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

def format_arabic(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def diacritize(text):
    global current_model, current_processor
    if not text.strip():
        return ""
    
    start_time = time.time()
    input_text = text.strip()
    
    # Preprocessing
    input_text = remove_tatweel(input_text)
    
    # Encode and process
    encoded_input = current_processor.encode_input(input_text)
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = current_model(input_tensor)
        _, pred_indices = torch.max(outputs, dim=2)
    
    # Post-processing
    pred_diacritics = current_processor.decode_diacritics(pred_indices[0].cpu().numpy()[:len(input_text)])
    diacritized_text = current_processor.apply_diacritics(input_text, pred_diacritics)
    
    # Clean up unwanted diacritics
    diacritized_text = remove_tatweel(diacritized_text)
    diacritized_text = remove_harakat_after_special_chars(diacritized_text)
    
    return diacritized_text, f"Processing time: {(time.time() - start_time)*1000:.2f} ms"

def process_file(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    diacritized, time_taken = diacritize(content)
    return diacritized, time_taken

available_models = list_available_models()
model_choices = [m[0] for m in available_models] if available_models else []
model_paths = {m[0]: (m[1], m[2]) for m in available_models}

with gr.Blocks(title="Arabic Diacritization", css=".arabic-text {direction: rtl; text-align: right; font-family: 'Arial'}") as demo:
    gr.Markdown("# Arabic Text Diacritization Tool")
    
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=model_choices,
            label="Select Model",
            value=model_choices[0] if model_choices else None
        )
        load_status = gr.Textbox(label="Model Status", interactive=False)
    
    with gr.Tab("Text Input"):
        text_input = gr.Textbox(label="Input Text", elem_classes=["arabic-text"])
        text_output = gr.Textbox(label="Diacritized Text", elem_classes=["arabic-text"])
        time_text = gr.Textbox(label="Processing Time")
        text_button = gr.Button("Diacritize Text")
    
    with gr.Tab("File Input"):
        file_input = gr.File(label="Upload Text File")
        file_output = gr.Textbox(label="Diacritized Content", elem_classes=["arabic-text"])
        file_time = gr.Textbox(label="Processing Time")
        file_button = gr.Button("Process File")
    
    def load_selected_model(model_name):
        if model_name in model_paths:
            model_path, processor_path = model_paths[model_name]
            return load_model(model_path, processor_path)
        return "Model not found!"
    
    model_selector.change(
        load_selected_model,
        inputs=model_selector,
        outputs=load_status
    )
    
    if model_choices:
        load_selected_model(model_choices[0])
    
    text_button.click(
        diacritize,
        inputs=text_input,
        outputs=[text_output, time_text]
    )
    
    file_button.click(
        process_file,
        inputs=file_input,
        outputs=[file_output, file_time]
    )

if __name__ == "__main__":
    demo.launch()