from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier

import numpy as np


class AdaBoost :
    """
    self.n nombre de modèle différent (ici des arbre de decision)->hyperparamètre
    self.apprentissage la valeur du taux d'apprentissage pour l'algorithme de descente de gradient de la méthode
    self.k -> valeur k à utiliser pour l'application de notre validation croisée de type 'k-fold'
    """
    def __init__(self,k):
        self.n =10
        self.apprentissage =0.01
        self.k= k


    def validation_croisee(self,X,T):
        """
        Méthode de validation croisée de type "k-fold" avec k = self.k


        :param X: matrice de données
        :param T: vecteur de cible
        :return: taux de réussite de classification sur l'ensemble des entrainements/tests effectués
        """

        kf = KFold(n_splits=self.k)
        kf.get_n_splits(X)


        classifieur = AdaBoostClassifier(n_estimators=self.n, learning_rate=self.apprentissage)
        score =0
        for index_entrainement, index_test in kf.split(X):
            x_entrainement, x_test, t_entrainement, t_test = X[index_entrainement], X[index_test], T[index_entrainement], T[index_test]
            classifieur.fit(x_entrainement, t_entrainement)
            score += classifieur.score(x_test, t_test)

        return score/self.k


    def hyperparametrage(self ,X ,T):

        n_estimateurs = [10, 50, 100, 300, 400]
        taux_apprentissage = [0.01, 0.05, 0.1, 1]

        n = self.n
        apprentissage = self.apprentissage

        meilleur_score = 0
        for i in n_estimateurs:
            self.n = i
            for j in taux_apprentissage:
                self.apprentissage = j
                score = self.validation_croisee(X, T)
                if score > meilleur_score:
                    meilleur_score = score
                    n = self.n
                    apprentissage = self.apprentissage



        self.n = n
        self.apprentissage = apprentissage

        print("Meilleur score: {:.3%}".format(meilleur_score))
        print("les meilleurs parametres sont :",self.n, self.apprentissage)
