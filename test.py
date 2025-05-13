from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Function to split text into manageable chunks with padding


def chunk_text_with_padding(text, max_length=512):
    # Tokenize the text and split into manageable chunks
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Check if adding this word exceeds max_length
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for the space
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    # Add the last chunk if there's any leftover
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":

    # Input text for testing
    text = "لقد قلت لي ذات يوم ونحن على جبل شلانجنبرج، إنك مستعد، بكلمة واحدة مني أن تلقي بنفسك إلى تحت، منكس الرأس، بينما نحن على علو ألف قدم. لسوف أقول هذه الكلمة يوما، لا لشيء إلا لأرى أأنت تقدم على التنفيذ حقا؛"

    # Model 1: Fine-Tashkeel model
    model_name_1 = "basharalrfooh/Fine-Tashkeel"
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_name_1)

    # Process text with Model 1 (Fine-Tashkeel)
    chunks = chunk_text_with_padding(
        text, max_length=512)  # Split the text into chunks
    output_1 = ""
    for chunk in chunks:
        input_ids_1 = tokenizer_1(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        outputs_1 = model_1.generate(input_ids_1, max_new_tokens=128)
        decoded_output_1 = tokenizer_1.decode(
            outputs_1[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_1 += decoded_output_1 + " "  # Concatenate the results

    # Model 2: Mushkil model using pipeline
    pipe_2 = pipeline("text2text-generation", model="riotu-lab/mushkil")
    output_2 = ""
    for chunk in chunks:
        decoded_output_2 = pipe_2(chunk)[0]['generated_text']
        output_2 += decoded_output_2 + " "  # Concatenate the results

    # Model 3: Byt5 Arabic Diacritization model
    pipe_3 = pipeline("text2text-generation",
                      model="glonor/byt5-arabic-diacritization")
    output_3 = ""
    for chunk in chunks:
        decoded_output_3 = pipe_3(chunk)[0]['generated_text']
        output_3 += decoded_output_3 + " "  # Concatenate the results

    # Output file to save results
    output_file = "diacritization_results.txt"

    # Write results to the file
    with open(output_file, "w", encoding="utf-8") as f:
        # Write original text
        f.write("Original Text:\n")
        f.write(text + "\n\n")

        # Write results from Model 1 (Fine-Tashkeel)
        f.write("Model 1 (Fine-Tashkeel) Output:\n")
        f.write(output_1 + "\n\n")

        # Write results from Model 2 (Mushkil)
        f.write("Model 2 (Mushkil) Output:\n")
        f.write(output_2 + "\n\n")

        # Write results from Model 3 (Byt5 Arabic Diacritization)
        f.write("Model 3 (Byt5 Arabic Diacritization) Output:\n")
        f.write(output_3 + "\n\n")

    print(f"Results have been saved to {output_file}")
