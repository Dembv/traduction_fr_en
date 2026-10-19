# Tr'App : Traduction automatique Français → Anglais

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

Projet de travaux pratiques (TP4) de traitement automatique du langage : un modèle **séquence-à-séquence à base de LSTM** entraîné sur un corpus de paires de phrases français/anglais, évalué avec le **score BLEU**, et exposé dans une petite interface web **Streamlit** (« Tr'App »).

---

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `tp4_traduction_en_fr.py` | Script complet du TP : chargement et nettoyage du corpus, tokenisation, entraînement du modèle, comparaison des prédictions, calcul et graphiques du score BLEU. |
| `home.py` | Interface Streamlit : l'utilisateur saisit une phrase en français et obtient sa traduction en anglais. |

## Fonctionnement

1. **Données** : le fichier `dataset/eng-french.csv` contient deux colonnes, `English words/sentences` et `French words/sentences`.
2. **Nettoyage** : mise en minuscules, suppression de la ponctuation (y compris `« »`), des chiffres et des espaces insécables.
3. **Découpage** : mélange aléatoire (`random_state=0`), puis séparation entraînement / test (80 / 20 dans le script, 90 / 10 dans l'application).
4. **Tokenisation** : un `Tokenizer` Keras par langue, séquences complétées par des zéros (*padding*) à la longueur de la phrase la plus longue.
5. **Modèle** (encodeur-décodeur) :

   ```
   Embedding (256, mask_zero) → LSTM (256)         ← encodeur
   RepeatVector (longueur cible)
   LSTM (256, return_sequences) → TimeDistributed(Dense softmax)   ← décodeur
   ```

   Optimiseur `adam`, perte `categorical_crossentropy`, jusqu'à 200 époques avec `EarlyStopping` (patience 10, meilleurs poids restaurés).
6. **Évaluation** : affichage côte à côte *source / cible attendue / traduction*, puis score BLEU cumulé de 1 à 4-grammes sur les jeux d'entraînement et de test (NLTK `corpus_bleu`).

## Installation

```bash
git clone https://github.com/Dembv/traduction_fr_en.git
cd traduction_fr_en
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install "tensorflow<2.16" nltk pandas numpy matplotlib ipython streamlit
```

> Le code utilise `keras.preprocessing.text.Tokenizer`, retiré dans Keras 3. Il faut donc une version de TensorFlow qui embarque encore Keras 2 (TensorFlow < 2.16).

Placez ensuite le corpus dans `dataset/eng-french.csv` (non versionné). Un jeu de données public de paires anglais-français, avec les deux colonnes ci-dessus, est disponible sur Kaggle sous le nom *English-French dataset*.

## Utilisation

**Script du TP** (entraînement sur 500 phrases, courbes de perte et scores BLEU) :

```bash
python tp4_traduction_en_fr.py
```

**Application Streamlit** (entraînement sur 5 000 phrases au lancement) :

```bash
streamlit run home.py
```

## Limites connues

- `home.py` importe un module `traduction` (fonctions `charger_dataset`, `creer_modele`, `predire_sequence`…) qui n'est pas présent dans le dépôt : il faut l'ajouter (ou le reconstituer à partir de `tp4_traduction_en_fr.py`) pour lancer l'application.
- Le modèle est réentraîné à chaque démarrage de l'application ; le sauvegarder (`model.save`) puis le recharger rendrait le lancement instantané.
- Avec quelques milliers de phrases et un vocabulaire limité, la qualité reste celle d'un modèle pédagogique : les mots absents du corpus d'entraînement ne peuvent pas être traduits.

## Auteur

Demba Hamet Guissé
