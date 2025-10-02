
import os
import sys
import codecs
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizerFast

def translate_text(text, model, tokenizer, src_lang="nep_Npi", target_lang="eng_Latn"):
    """
    Translates a single text string.
    """
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.vocab[target_lang],
            max_length=512
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        return f"An error occurred during translation: {e}"

def main():
    """
    Main function to load the model and run a test translation.
    """
    # Reconfigure stdout to handle UTF-8 encoding
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

    # --- Configuration ---
    # Construct the absolute path to the model directory to ensure it's found correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "nllb-finetuned-nepali-en")
    
    # --- Model Loading ---
    print("Loading model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = NllbTokenizerFast.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # --- Translation ---
    sentences_to_translate = [
        "मेरो नाम जेमिनी हो।",
        "आज मौसम कस्तो छ?",
        "मलाई नेपाली खाना मन पर्छ।",
        "तपाईंलाई कस्तो छ?"
    ]

    for sentence in sentences_to_translate:
        print(f"\nOriginal text (Nepali): '{sentence}'")
        translated_text = translate_text(sentence, model, tokenizer)
        print(f"Translated text (English): '{translated_text}'")


if __name__ == "__main__":
    main()
