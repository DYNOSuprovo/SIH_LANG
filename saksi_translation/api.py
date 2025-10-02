# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, NllbTokenizerFast
import torch

app = FastAPI(title="Saksi Translation API", version="2.0")

# Device: CPU for demo; change to "cuda" if GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models (CPU-friendly demo)
models = {
    "nepali": MBartForConditionalGeneration.from_pretrained(
        "models/nllb-finetuned-nepali-en"
    ).to(DEVICE),
    # Uncomment when Sinhalese model is available
    # "sinhalese": MBartForConditionalGeneration.from_pretrained(
    #     "models/nllb-finetuned-sinhalese-en"
    # ).to(DEVICE),
}

tokenizers = {
    "nepali": NllbTokenizerFast.from_pretrained("models/nllb-finetuned-nepali-en"),
    # "sinhalese": NllbTokenizerFast.from_pretrained("models/nllb-finetuned-sinhalese-en"),
}

# Pydantic request model
class TranslationRequest(BaseModel):
    text: str
    source_language: str  # "nepali" or "sinhalese"

@app.post("/translate")
def translate(req: TranslationRequest):
    src_lang = req.source_language.lower()
    text = req.text.strip()

    if src_lang not in models:
        raise HTTPException(status_code=400, detail=f"Language '{src_lang}' not supported.")

    model = models[src_lang]
    tokenizer = tokenizers[src_lang]

    # Target language code: English (Latin script)
    tgt_code = "eng_Latn"

    try:
        inputs = tokenizer(text, return_tensors="pt")
        # Use get_lang_id() instead of lang_code_to_id
        inputs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_code)

        outputs = model.generate(**inputs, max_length=512)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"translation": translated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Saksi Translation API is running!"}
