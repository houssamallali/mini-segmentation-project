# Analyse de l'ouverture d'un fusible haute capacité de rupture

Ce projet implémente un système d'analyse d'images pour mesurer l'évolution de la distance d'ouverture d'un fusible industriel à partir de séquences vidéo radiographiques.

## Description

L'analyse porte sur des fusibles haute capacité de rupture utilisés dans l'industrie. Lors de leur activation, un arc électrique est généré et le sable fond dans le voisinage, créant une séparation progressive entre les deux parties du fusible. Ce script mesure automatiquement cette distance d'ouverture au fil du temps.

## Fonctionnalités

- **Segmentation automatique** : Détection du corps du fusible par seuillage adaptatif d'Otsu
- **Calibration** : Étalonnage pixel/mètre basé sur la hauteur connue du fusible (2 mm)
- **Mesure temporelle** : Calcul de la distance d'ouverture pour chaque image de la séquence
- **Exportation** : Sauvegarde des résultats en CSV et génération d'un graphique

## Prérequis

```bash
pip install opencv-python numpy matplotlib scipy
```

## Utilisation

```bash
python fuse_gap.py --video Camera_15_04_58.mp4
```

Le script génère automatiquement :
- `distance.csv` : Données de mesure par image
- `distance_plot.png` : Graphique de l'évolution temporelle

## Méthode

1. **Prétraitement** : Flou gaussien et seuillage binaire inversé
2. **Nettoyage morphologique** : Opérations d'ouverture et fermeture
3. **Calibration** : Mesure de la hauteur H sur la première image
4. **Détection des composantes** : Identification des deux moitiés du fusible
5. **Calcul de l'écart** : Mesure de la distance horizontale entre les composantes

## Paramètres ajustables

Les paramètres principaux peuvent être modifiés en début de script :
- `BLUR_KERNEL_SIZE` : Taille du noyau de flou (défaut: 5)
- `MORPH_STRUCT_SIZE` : Taille de l'élément structurant (défaut: 3)
- `MIN_COMP_AREA` : Surface minimale des composantes (défaut: 500 pixels)

## Résultats

Le graphique généré reproduit la courbe caractéristique de l'ouverture du fusible, montrant l'évolution progressive de la distance d'écartement en fonction du numéro d'image.

## Contexte académique

Ce travail s'inscrit dans le cadre d'un mini-projet de traitement d'images, reproduisant une analyse initialement développée en MATLAB pour l'étude de fusibles industriels haute performance.
