import gc
import time
import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Garde en mémoire le dernier modèle chargé pour éviter de le recharger à chaque clic
cache_modele = {"cle": None, "processor": None, "model": None}

def generer_legende(image, model_id, mode):
    if image is None:
        return "Veuillez charger une image.", "-", "-"

    # Redimensionner pour protéger la mémoire (8 Go RTX 5060 / 12 Go Colab)
    image = image.convert("RGB")
    image.thumbnail((512, 512))

    cle_actuelle = f"{model_id}_{mode}"

    # Charger le modèle uniquement s'il a changé
    if cache_modele["cle"] != cle_actuelle:
        cache_modele["processor"] = None
        cache_modele["model"] = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if mode == "4bit":
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
        elif mode == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

        cache_modele["processor"] = AutoProcessor.from_pretrained(model_id)
        cache_modele["model"] = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            quantization_config=quant_config,
            device_map="auto"
        )
        cache_modele["cle"] = cle_actuelle

    processor = cache_modele["processor"]
    model = cache_modele["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image clearly in one or two sentences."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

    debut = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=60)
    duree = time.time() - debut

    reponse_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    legende = processor.batch_decode(reponse_ids, skip_special_tokens=True)[0].strip()
    vram_go = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return legende, f"{duree:.2f} secondes", f"{vram_go:.2f} Go"

# Interface graphique Gradio
interface = gr.Interface(
    fn=generer_legende,
    inputs=[
        gr.Image(type="pil", label="1. Charger une image"),
        gr.Dropdown(
            choices=["HuggingFaceTB/SmolVLM-Instruct", "Qwen/Qwen2-VL-2B-Instruct"],
            value="HuggingFaceTB/SmolVLM-Instruct",
            label="2. Choisir le modèle VLM"
        ),
        gr.Radio(
            choices=["fp16", "8bit", "4bit"],
            value="4bit",
            label="3. Niveau de quantification"
        )
    ],
    outputs=[
        gr.Textbox(label="Légende générée", lines=3),
        gr.Textbox(label="Temps d'inférence"),
        gr.Textbox(label="VRAM maximale utilisée")
    ],
    title="Projet 7 — Comparateur de VLMs quantifiés",
    description="Testez l'impact de la quantification (FP16, 8 bits, 4 bits) sur la qualité des légendes, la vitesse et la mémoire GPU."
)

if __name__ == "__main__":
    interface.launch(share=True)