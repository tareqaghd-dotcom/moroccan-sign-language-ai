# État du dataset WLASL100

## Situation initiale

Nous avons créé `asl_raw.csv` à partir de `WLASL100.json`.

Il contient :

- 100 classes
- 2038 instances attendues

## Problème rencontré

Toutes les vidéos WLASL ne sont plus disponibles.

Certaines vidéos YouTube sont :

- privées
- supprimées
- indisponibles
- associées à des URLs expirées

Après le téléchargement des vidéos brutes, nous avions 1024 fichiers dans
`raw_videos/`.

Après le preprocessing, nous avons obtenu actuellement 1137 clips `.mp4`
dans le dossier `videos/`.

Cependant, nous devons encore vérifier lesquels de ces clips correspondent
réellement aux instances présentes dans `asl_raw.csv`.

## Solution choisie

Nous conservons `asl_raw.csv` intact comme fichier de référence.

Nous allons créer :

`asl_selected.csv`

Ce fichier contiendra uniquement les instances de `asl_raw.csv` pour lesquelles
le fichier vidéo correspondant existe réellement.

Exemple :

asl_raw.csv :

12319,BOOK,train
12320,BOOK,train
12321,BOOK,val

Si :

12319.mp4 existe
12320.mp4 existe
12321.mp4 n'existe pas

Alors `asl_selected.csv` contiendra seulement :

12319,BOOK,train
12320,BOOK,train

## Ajout de nouvelles classes

Nous envisageons ensuite d'ajouter de nouvelles classes WLASL.

Une nouvelle instance sera ajoutée à notre dataset seulement si :

1. sa vidéo est accessible ;
2. le téléchargement réussit réellement ;
3. le clip final peut être généré.

Nous évitons ainsi d'ajouter dans le CSV des vidéos privées, supprimées ou
indisponibles.