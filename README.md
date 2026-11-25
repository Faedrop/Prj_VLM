# Projet 7 — Évaluation de petits modèles vision-langage (VLMs) quantifiés


**Cadre :** Projet d'IA Générative Visuelle — Option A (Génération automatique de légendes)  
**Matériel de test :** NVIDIA RTX 5060 (8 Go VRAM) — Compatible Google Colab / Kaggle ($\le 12\text{ Go}$ VRAM)

---

## 1. Présentation du projet

Ce projet étudie l'impact de la **quantification mémoire** (**FP16**, **8 bits** et **4 bits**) sur les performances de deux petits modèles vision-langage (**VLMs**) d'environ 2 milliards de paramètres :
* **`HuggingFaceTB/SmolVLM-Instruct`** (2,2B paramètres)
* **`Qwen/Qwen2-VL-2B-Instruct`** (2B paramètres)

L'objectif est de mesurer le compromis entre **consommation mémoire (VRAM)**, **vitesse d'inférence** et **qualité des légendes générées** (métrique **ROUGE-L**) sur un sous-ensemble fixe de **500 images de Flickr8k** (chacune associée à 5 légendes humaines de référence), sans réentraînement complet des modèles.

---

## 2. Structure du dépôt

```text
projet7-vlm/
├── README.md                              # Présentation, installation et résultats principaux
├── requirements.txt                       # Dépendances Python du projet
├── configs/
│   └── config.yaml                        # Paramètres centralisés de l'expérience
├── data/
│   ├── prepare_data.py                    # Téléchargement et échantillonnage de 500 images Flickr8k
│   └── flickr8k_500.json                  # Index léger des images et des 5 légendes de référence
├── src/
│   ├── premier_test.py                    # Test unitaire sur 1 image (SmolVLM)
│   ├── test_qwen.py                       # Test unitaire sur 1 image (Qwen2-VL-2B)
│   ├── evaluer.py                         # Benchmark complet (2 modèles × 3 précisions sur 500 images)
│   ├── graphiques.py                      # Génération des courbes VRAM et ROUGE-L
│   └── exemples_visuels.py                # Création de la planche comparative image + légendes
├── notebooks/
│   ├── 01_donnees.ipynb                   # Exploration des données Flickr8k
│   ├── 02_baseline_fp16.ipynb             # Référence en précision normale (FP16)
│   ├── 03_quantification_8bit_4bit.ipynb  # Comparaison des compressions 8 bits et 4 bits
│   ├── 04_evaluation_metriques.ipynb      # Analyse des résultats chiffrés et graphiques (500 images)
│   └── 05_demo.ipynb                      # Notebook de démonstration interactive (compatible Colab)
├── results/
│   ├── resume_metriques.csv               # Tableau récapitulatif des performances
│   ├── details_legendes.csv               # Ensemble des 3 000 légendes générées
│   ├── graphique_vram.png                 # Comparaison de la consommation VRAM
│   ├── graphique_qualite.png              # Comparaison des scores ROUGE-L
│   └── exemples_visuels.png               # Exemples qualitatifs côte à côte
├── demo/
│   ├── app.py                             # Application web interactive Gradio
├── docs/
│   ├── fiche_modeles_et_licences.md       # Caractéristiques techniques et licences
│   └── reflexion_ethique.md               # Limites, biais, hallucinations et sobriété numérique
└── report/
    └── rapport_projet7.pdf              