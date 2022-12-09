from sklearn.ensemble import AdaBoostClassifier
from Méthodes.AdaBoost import Entrainement_AdaBoost as sab
import pandas as pd



def adaboost_main(k,normalisation):
    # création du gestionnaire des données et de notre objet solution_random_forest
    gestionnaire_donnees = gestion_donnees.GestionDonnees(normalisation)
    npl = sab.Solution_AdaBoost(k)

    # jeu de données d'entrainement sous forme de matrice
    X = gestionnaire_donnees.X_entrainement
    T = gestionnaire_donnees.T_entrainement

    # Recherche des hyperparametres
    npl.hyperparametrage(X, T)

    # données de Test sous forme de matrice
    X_test = gestionnaire_donnees.X_test

    # Entrainement du modèle
    classifieur = AdaBoostClassifier(n_estimators=npl.n, learning_rate=npl.apprentissage)
    classifieur.fit(X, T)

# A compléter

