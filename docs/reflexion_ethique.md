# Réflexion éthique et limites

## 1. Hallucinations et perte de précision en 4 bits
La quantification en 4 bits réduit fortement la mémoire GPU requise, mais peut amener le modèle à simplifier excessivement la scène ou à inventer des détails absents de l'image (hallucinations). Pour entraîner une LoRA, une vérification humaine des légendes générées reste indispensable.

## 2. Biais hérités des données
Les VLMs et le jeu Flickr8k reflètent principalement des scènes et des contextes culturels occidentaux. Le modèle peut mal interpréter des objets, tenues ou architectures spécifiques à d'autres régions (comme le patrimoine marocain) ou reproduire des stéréotypes de genre dans ses descriptions.

## 3. Impact environnemental (Sobriété numérique)
L'utilisation de petits modèles (2B paramètres) quantifiés en 4 bits divise par plus de 2 la consommation de mémoire VRAM (< 2 Go) par rapport au FP16. Cela permet d'exécuter l'IA localement sur des cartes grand public sans recourir à de lourds serveurs énergivores.

## 4. Propriété intellectuelle (Option A — Préparation de LoRA)
Lorsque le VLM sert à légender automatiquement des images pour entraîner une LoRA, il faut s'assurer que les images sources sont libres de droits ou sous licence compatible (ex. Wikimedia Commons) et ne pas copier le style d'artistes vivants sans autorisation.