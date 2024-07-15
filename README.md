# test-technique-vif

Projet pour un test technique à réaliser dans le cadre d'un processus de recrutement pour la société Vif.   

## Instructions 

### Objectif de l'exercice :
Construire un modèle ML pour prédire si la condition valve est optimale (=100%) ou non, pour chaque cycle.   
Vous utiliserez les 2000 premiers cycles pour construire le modèle et le reste comme échantillon de test final.   

### Données :
Les données sont téléchargeables avec les commandes suivantes :   
  - Pour Windows :   
    - _New-Item -ItemType Directory -Force -Path "data"_   
    - _Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip" -OutFile "data\data.zip"_   
    - _Expand-Archive -Path "data\data.zip" -DestinationPath "data"_   

Les données sont décrites dans le lien suivant : https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems.   
Le dossier en pièce jointe à utiliser contient uniquement les 3 fichiers suivants, chaque ligne représentant un cycle : 
  - PS2 (Pression (bar) echantillonnage 100Hz) 
  - FS1 (Volume flow (l/min) echantillonnage 10Hz) 
  - Profile : Fichier avec les variables dont la "valve condition" qui nous intéresse.   

### Bonus :
Voici quelques points bonus que vous pouvez choisir de réaliser en plus :
  - Mettre votre solution sur github ou gitlab 
  - Ajouter des tests unitaires 
  - Containeriser votre code pour qu'il puisse être exécuté facilement par un tiers 
  - Mettre en place une application web qui donne la prédiction pour un numéro de cycle donné en entrée (cf. test-technique-vif-api-dashboard)

## ENTRAÎNEMENT DU MODÈLE

Ce projet permet de charger les données et d'entraîner un modèle de classification.   
Le modèle utilisé est un RNN avec une couche GRU (Gated-Recurrent-Unit) construit à partir de la classe Sequential() de Keras.   
Différents paramètres ont été testé :   
- Layer : "gru" ou "lstm" (Couche GRU ou LSTM)
- Resampling : "up" ou "down" (harmonisation des fréquences entre FS1 et PS2)
- Shuffle : "True" or "False"
- Train_size : 2000 (maintien du choix demandé dans les objectifs, mais possibilité dans le code de modifier ce paramètre)
- Normalize : "True" or "False" (Normalisation des données ou non avec un StandardScaler() sklearn)

Les paramètres retenus sont :   
```python
{'layer': "gru", 'resampling': 'up', 'shuffle': True, 'train_size': 2000, 'normalize': False}
```
La Matrice de confusion a été tracée :   
![Matrice de confusion](test_technique_vif/Matrice_de_confusion__Model_Sequential__Layer_gru__Resampling_up__Scaling_no_scaler__Shuffling_shuffled.png)

## PROJETS GIT

### ENTRAÎNEMENT DU MODÈLE

https://github.com/zerippeur/test-technique-vif   

### API ET DASHBOARD

https://github.com/zerippeur/test-technique-vif-api-dashboard   