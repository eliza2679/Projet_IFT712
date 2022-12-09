from sklearn.ensemble import AdaBoostClassifier
from Méthodes.AdaBoost import Entrainement_AdaBoost as eab
import pandas as pd
from gestion_donnees import gestion_donnees




def adaboost_main(k,normalisation):
    # création du gestionnaire des données et de notre objet solution_random_forest
    gestionnaire_donnees = gestion_donnees.GestionDonnees(normalisation)
    npl = eab.Solution_AdaBoost(k)

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

 # Prédiction (sous forme de probabilités d'appartenance à chaque classe) sur les données de test
    resultat_test = classifieur.predict_proba(X_test)

    # Mise des resulats sous un format csv 
    resultat = pd.DataFrame(resultat_test, columns=gestionnaire_donnees.encodeur.classes_)
    resultat.insert(0, 'id', gestionnaire_donnees.id_test)
    resultat.reset_index()

    if gestionnaire_donnees.normal :
        resultat.to_csv('resultats/resultatAdaBoost_normalise.csv', index=False)
    else :
        resultat.to_csv('resultats/resultatAdaBoost.csv', index=False)

