# CLAUD·IA — Interprète vocal trilingue

Travail pratique de Machine Learning. Le projet traduit de courtes phrases
parlées entre le français, l’anglais et l’espagnol et produit un audio que
l’on peut écouter.

## Programme prévu

Le notebook Python utilisera le Transformer préentraîné SeamlessM4T v2.
Il chargera un extrait audio, le préparera en mono à 16 kHz, sélectionnera
la langue cible et fera écouter la traduction produite.

L’ajustement fin supervisé portera sur la traduction de la parole espagnole
vers le texte anglais. La synthèse vocale utilisera les paramètres préentraînés.
Le français fera partie de la démonstration avec le modèle préentraîné.

## Données et évaluation


**FLEURS** signifie *Few-shot Learning Evaluation of Universal Representations of Speech*. Ce jeu de données publié par Google contient des enregistrements de phrases équivalentes dans plusieurs langues. Pour l’ajustement fin de notre Transformer, chaque exemple associe un enregistrement en espagnol à sa traduction écrite en anglais. [Source des données](https://huggingface.co/datasets/google/fleurs)

- **Entraînement et validation** — Enregistrements en espagnol et traductions
  écrites correspondantes en anglais, issus des partitions distinctes de FLEURS.
- **Test final** — Six phrases originales en espagnol enregistrées à voix
  normale et à voix basse, soit douze audios. Les mêmes fichiers serviront à
  comparer le modèle préentraîné et le modèle ajusté.
- **Mesures** — Fidélité du sens par rapport à une traduction de référence
  et temps médian de génération de l’audio après chargement du modèle.

## Livrables

- [ ] Notebook de démonstration et instructions d’exécution.
- [ ] Code d’ajustement fin et protocole reproductible.
- [ ] Exemple de démonstration, source des données et résultats mesurés.
- [ ] Diapositives et synthèse d’une à deux pages.

Le notebook, l’ajustement fin et les mesures sont à réaliser.

## Sources

- [SeamlessM4T v2 — Meta](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [FLEURS — Google](https://huggingface.co/datasets/google/fleurs)
- [Guide d’ajustement fin — Meta](https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/cli/m4t/finetune)
