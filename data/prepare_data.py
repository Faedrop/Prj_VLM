import os
import json
from datasets import load_dataset


os.makedirs("data/images", exist_ok=True)

print("Téléchargement de 500 images depuis Flickr8k...")
# On charge uniquement les 500 premières images du jeu de test
dataset = load_dataset("jxie/flickr8k", split="test[:500]")

donnees_propres = []

for i, exemple in enumerate(dataset):

    nom_fichier = f"img_{i:03d}.jpg"
    chemin_image = os.path.join("data/images", nom_fichier)
    

    exemple["image"].convert("RGB").save(chemin_image)
    
    # Récupération des 5 légendes humaines de référence
    legendes = [exemple[f"caption_{j}"] for j in range(5)]
    
    donnees_propres.append({
        "id": i,
        "image_path": chemin_image,
        "references": legendes
    })

with open("data/flickr8k_500.json", "w", encoding="utf-8") as f:
    json.dump(donnees_propres, f, indent=2, ensure_ascii=False)

print("Terminé ! 500 images prêtes dans data/images/ et indexées dans data/flickr8k_500.json.")