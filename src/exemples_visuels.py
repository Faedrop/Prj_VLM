import csv
import textwrap
import matplotlib.pyplot as plt
from PIL import Image

# On lit les légendes générées lors de ton évaluation
lignes = []
with open("results/details_legendes.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lignes.append(row)

# On choisit 3 images parmi les 50 traitées (ex: images 0, 1 et 2)
ids_choisis = ["0", "1", "2"]

fig, axes = plt.subplots(len(ids_choisis), 2, figsize=(13, 11), gridspec_kw={"width_ratios": [1, 2.3]})

for idx, img_id in enumerate(ids_choisis):
    # Afficher la photo à gauche
    img = Image.open(f"data/images/img_{int(img_id):03d}.jpg")
    axes[idx, 0].imshow(img)
    axes[idx, 0].axis("off")
    axes[idx, 0].set_title(f"Image #{img_id}", fontsize=11, fontweight="bold")

    # Récupérer les différentes légendes pour cette image
    sous_lignes = [r for r in lignes if r["image_id"] == img_id]
    ref = sous_lignes[0]["reference_1"]
    
    texte_comparatif = f"Référence humaine : {ref}\n\n"
    for r in sous_lignes:
        if r["precision"] in ["fp16", "4bit"]:  # On compare normal vs 4 bits
            nom = r["modele"].split("-")[0]
            prec = r["precision"].upper()
            texte_comparatif += f"[{nom} - {prec}] : {r['legende_generee']}\n"

    # Retour à la ligne automatique pour que le texte reste lisible
    lignes_formatees = "\n".join(textwrap.fill(p, width=85) for p in texte_comparatif.split("\n"))
    
    axes[idx, 1].text(0, 0.5, lignes_formatees, va="center", ha="left", fontsize=9.5, family="sans-serif")
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.savefig("results/exemples_visuels.png", dpi=200)
plt.close()
print("Planche visuelle enregistrée dans results/exemples_visuels.png !")