import csv
import matplotlib.pyplot as plt

# 1. Lire les résultats générés par evaluer.py
lignes = []
with open("results/resume_metriques.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lignes.append(row)

modeles = ["SmolVLM-Instruct", "Qwen2-VL-2B-Instruct"]
precisions = ["fp16", "8bit", "4bit"]
x = [0, 1, 2]
largeur = 0.35

def extraire_valeurs(nom_modele, colonne):
    valeurs = []
    for p in precisions:
        for row in lignes:
            if row["modele"] == nom_modele and row["precision"] == p:
                valeurs.append(float(row[colonne]))
    return valeurs

# --- Graphique 1 : Consommation mémoire (VRAM) ---
vram_smol = extraire_valeurs("SmolVLM-Instruct", "vram_max_go")
vram_qwen = extraire_valeurs("Qwen2-VL-2B-Instruct", "vram_max_go")

plt.figure(figsize=(8, 5))
plt.bar([i - largeur/2 for i in x], vram_smol, width=largeur, label="SmolVLM-Instruct", color="#4C72B0")
plt.bar([i + largeur/2 for i in x], vram_qwen, width=largeur, label="Qwen2-VL-2B", color="#55A868")
plt.xticks(x, ["FP16 (Normal)", "8 bits", "4 bits"])
plt.ylabel("VRAM maximale (Go)")
plt.title("Impact de la quantification sur la mémoire GPU (VRAM)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/graphique_vram.png", dpi=200)
plt.close()

# --- Graphique 2 : Qualité des légendes (Score ROUGE-L) ---
rouge_smol = extraire_valeurs("SmolVLM-Instruct", "rouge_l")
rouge_qwen = extraire_valeurs("Qwen2-VL-2B-Instruct", "rouge_l")

plt.figure(figsize=(8, 5))
plt.bar([i - largeur/2 for i in x], rouge_smol, width=largeur, label="SmolVLM-Instruct", color="#4C72B0")
plt.bar([i + largeur/2 for i in x], rouge_qwen, width=largeur, label="Qwen2-VL-2B", color="#55A868")
plt.xticks(x, ["FP16 (Normal)", "8 bits", "4 bits"])
plt.ylabel("Score ROUGE-L (plus haut = meilleur)")
plt.title("Qualité des légendes selon le niveau de quantification")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/graphique_qualite.png", dpi=200)
plt.close()

print("Graphiques enregistrés dans results/graphique_vram.png et results/graphique_qualite.png !")