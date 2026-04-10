# NHL_Play_By_Play
 
 ## Description du projet

 Ce projet a pour objectif principal de prédire, à partir de données provenant de la Ligue nationale de hockey (LNH), quels lancers se convertiront en buts. 

 Il combine **l'ingénierie et la visualisation des données**, le stockage hybride avec **SQLite et Redis**, et l'utilisation de modèles d'**apprentissage automatique**, documentées avec **WandB**. 

  Une **analyse avancée avec visualisation des données** (À VENIR) est l'objectif secondaire du projet. 
 
 Une exploration préliminaire des données sous forme de cahiers Jupyter est aussi incluse. 

 ## Architecture

 Le pipeline d'exécution télécharge et transforme les données "play-by-play" de format JSON provenant de l'API de la LNH. 
 
* **Ingestion des données** : `download_data.py` (saisons 2020-2025).

* **Stockage** : SQLite pour la persistance, Redis pour la mise en cache et pour un accès rapide aux paramètres préliminaires nécessaires aux modèles d'apprentissage. 

* **Modélisation** : Modèles de **régression logistique, Random Forest et XGBoost**, avec ingénierie des données et division en ensembles d'entrainement et de test via `modeling_utils.py`.  

L'exploration préliminaire des données se trouve dans le dossier *notebooks*, puis l'analyse détaillée (À VENIR) se trouve dans le dossier *pipeline/analytics*.

## Technologies utilisées 

**Machine Learning** : Scikit-learn, XGBoost, WandB (tracking d'expériences)

**Gestion des données** : Pandas, SQLite, Redis

**Reproductibilité** : Docker

 ## Comment exécuter le projet

 Le projet nécessite **Docker Desktop**, et optionnellement un compte **WandB** pour suivre les expériences. 

 1. **Cloner le projet** : `git clone https://github.com/Frigonf1/NHL_Play_By_Play.git`

 2. Créer un ficher `.env` avec votre clé API WandB appelée **WANDB_API_KEY**. 

 3. **Lancement de Docker** : 

 ``` bash
docker compose up -d
docker ps
 ```

Le conteneur exécutera automatiquement le script `main.py`, qui téléchargera les données et créera la base de données. Le téléchargement pourrait prendre plusieurs minutes. 

Les modèles d'apprentissage automatique peuvent être exécutés individuellement à partir du fichier *pipeline/modeling*. 

*Prochaines étapes : analyse visuelle des données avec dashboard interactif.*
