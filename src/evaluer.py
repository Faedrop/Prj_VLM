import os
import gc
import csv
import json
import time
import torch
import evaluate
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Paramètres de l'expérience
MODELES = [
    "HuggingFaceTB/SmolVLM-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct"
]
MODES = ["fp16", "8bit", "4bit"]

# Commence avec 50 images (~5 min au total) pour vérifier, puis passe à 500 pour le résultat final
NB_IMAGES = 500
PROMPT_TEXTE = "Describe this image clearly in one or two sentences (subject, action, background, colors)."

def charger_modele(model_id, mode):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if mode == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif mode == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto"
    )
    return processor, model

def lancer_evaluation():
    os.makedirs("results", exist_ok=True)
    rouge = evaluate.load("rouge")

    with open("data/flickr8k_500.json", "r", encoding="utf-8") as f:
        donnees = json.load(f)[:NB_IMAGES]

    resume_global = []
    details_legendes = []

    for model_id in MODELES:
        nom_court = model_id.split("/")[-1]

        for mode in MODES:
            print(f"\n=== Évaluation : {nom_court} | Mode : {mode.upper()} ({NB_IMAGES} images) ===")
            processor, model = charger_modele(model_id, mode)

            predictions = []
            references = []
            temps_liste = []

            for i, item in enumerate(donnees):
                image = Image.open(item["image_path"]).convert("RGB")
                # Redimensionnement uniforme (recommandé par le guide)
                image.thumbnail((512, 512))

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": PROMPT_TEXTE}
                        ]
                    }
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

                debut = time.time()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=50)
                duree = time.time() - debut

                reponse_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                legende = processor.batch_decode(reponse_ids, skip_special_tokens=True)[0].strip()

                predictions.append(legende)
                references.append(item["references"])
                temps_liste.append(duree)

                details_legendes.append({
                    "image_id": item["id"],
                    "modele": nom_court,
                    "precision": mode,
                    "legende_generee": legende,
                    "reference_1": item["references"][0]
                })

                if (i + 1) % 10 == 0:
                    print(f"  Progression : {i + 1}/{NB_IMAGES} images traitées...")

            # Mesures finales pour cette configuration
            vram_go = torch.cuda.max_memory_allocated() / (1024 ** 3)
            temps_moyen = sum(temps_liste) / len(temps_liste)
            score_rouge = rouge.compute(predictions=predictions, references=references)["rougeL"]

            print(f"--> Résultat {nom_court} ({mode}) : VRAM = {vram_go:.2f} Go | Temps = {temps_moyen:.2f}s/img | ROUGE-L = {score_rouge:.4f}")

            resume_global.append({
                "modele": nom_court,
                "precision": mode,
                "nb_images": NB_IMAGES,
                "vram_max_go": round(vram_go, 2),
                "temps_moyen_sec": round(temps_moyen, 2),
                "rouge_l": round(score_rouge, 4)
            })

            del model, processor
            gc.collect()
            torch.cuda.empty_cache()

    # Sauvegarde du tableau récapitulatif dans results/
    with open("results/resume_metriques.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resume_global[0].keys())
        writer.writeheader()
        writer.writerows(resume_global)

    # Sauvegarde de toutes les légendes générées pour analyse qualitative
    with open("results/details_legendes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details_legendes[0].keys())
        writer.writeheader()
        writer.writerows(details_legendes)

    print("\nTerminé ! Les fichiers sont enregistrés dans le dossier results/.")

if __name__ == "__main__":
    lancer_evaluation()