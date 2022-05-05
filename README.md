# Prédiction de la demande d'électricité au Portgual : une approche d'apprentissage
Simon Thérien - Mag Energy Solutions

## 1. Mise en contexte
On cherche à prédire la demande d'électricité d'utilisateurs ménagers pour les 5 prochains jours (ou 120 prochaines heures) en prenant en entrée 720 heures d'utilisation. On dispose d'un [jeu de données](https://github.com/huggingface/datasets/tree/master/datasets/electricity_load_diagrams) composé de séries temporelles de consommation en kWh de 250 foyers portugais de 2011 à 2014. On fera appel à un modèle d'apprentissage automatique décrit dans l'article *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks* que l'on décrira plus en détails dans les sections suivantes. À haut niveau, le modèle appelé LSTNet utilise une combinaison de réseaux de neurones récurrents (utilisés en analyse du langage) et de réseaux de neurones convolutifs (utilisés en imagerie par ordinateur) pour extraire des tendances à court terme (tendances quotidiennes) et à long terme (tendances saisonnières) présentes dans des données de consommation d'électricité.

## 2. Données
Avant de s'attaquer à l'apprentissage statistique, il est utile de visualiser les données. Comme mentionné précédemment, on remarque de la saisonalité à court terme (e.g. périodes quotidiennes de pointe): ![alt text](viz_1.png)
et de la saisonalité à long terme (e.g. température au fil des saisons): ![alt text](viz_2.png)
Pour ce qui suit, nous nous concentrons sur la tendance à court terme car nous cherchons à faire des prédictions à court terme.

## 3. Modèlisation
### 3.1 Description

## Résultats

## Conclusion et pistes d'améliorations

