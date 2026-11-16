**Langues :** [Français](README.md) · [Español](README.es.md)

# CLAUD·IA — Interprète vocal trilingue

Travail pratique de Machine Learning. Le projet traduit de courtes phrases
parlées entre le français, l’anglais et l’espagnol et produit un audio que
l’on peut écouter.

## Programme

Le notebook Python utilise le Transformer préentraîné SeamlessM4T v2.
Il charge un extrait audio, le prépare en mono à 16 kHz, sélectionne
la langue cible et fait écouter la traduction produite.

Le fine-tuning supervisé pilote a porté sur la traduction de la parole
espagnole vers le texte anglais. La synthèse vocale utilise les paramètres
préentraînés. Le français est aussi testé avec le modèle préentraîné.

## Données et évaluation


**FLEURS** signifie *Few-shot Learning Evaluation of Universal Representations of Speech*. Ce jeu de données publié par Google contient des enregistrements de phrases équivalentes dans plusieurs langues. Pour l’ajustement fin de notre Transformer, chaque exemple associe un enregistrement en espagnol à sa traduction écrite en anglais. [Source des données](https://huggingface.co/datasets/google/fleurs)

- **Entraînement et validation** — Enregistrements en espagnol et traductions
  écrites correspondantes en anglais, issus des partitions distinctes de FLEURS.
- **Test final** — Six phrases originales en espagnol enregistrées à voix
  normale et à voix basse, soit douze audios. Les mêmes fichiers ont servi à
  comparer le modèle préentraîné et le modèle ajusté.
- **Résultat** — Pour chacune des deux versions du modèle, le sens a été
  conservé dans les six prises normales et dans cinq des six prises à voix
  basse. Un essai distinct a mesuré le temps de génération sur CPU et sur T4.
  [Protocole, résultats et captures](INSTALLATION_ET_DEPANNAGE.md)

## Livrables

- [x] Notebook de démonstration et instructions d’exécution.
- [x] Code de fine-tuning et protocole du pilote.
- [x] Exemple de démonstration, source des données et résultats mesurés.
- [ ] Diapositives et synthèse d’une à deux pages.

## Sources

- [SeamlessM4T v2 — Meta](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [FLEURS — Google](https://huggingface.co/datasets/google/fleurs)
- [Guide fine-tuning — Meta](https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/cli/m4t/finetune)
