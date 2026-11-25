import time
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
IMAGE_PATH = "data/images/img_000.jpg"

def tester_vlm(mode="fp16"):
    print(f"\n--- Test en mode : {mode.upper()} ---")
    
    # Nettoyer la mémoire GPU avant chaque test
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Configurer la quantification si demandée
    if mode == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif mode == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    # 2. Charger le processeur et le modèle
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto"
    )

    # 3. Préparer l'image et la consigne (Option A : légende détaillée pour LoRA)
    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail (subject, background, colors, lighting)."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    # 4. Générer la légende et chronométrer
    debut = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80)
    duree = time.time() - debut

    # Extraire uniquement la réponse générée (sans répéter le prompt)
    reponse_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    legende = processor.batch_decode(reponse_ids, skip_special_tokens=True)[0].strip()

    # 5. Mesurer le pic de VRAM en Gigaoctets (Go)
    vram_go = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"Légende : {legende}")
    print(f"Temps   : {duree:.2f} secondes")
    print(f"VRAM    : {vram_go:.2f} Go")

    # Libérer le modèle de la mémoire avant le test suivant
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    tester_vlm(mode="fp16")
    tester_vlm(mode="8bit")
    tester_vlm(mode="4bit")