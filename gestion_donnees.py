# Classe qui permet de transposer les données du format csv en matrice numpy

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


class GestionDonnees:

    def __init__(self, normal=False):
        self.normal = normal
        df_entrainement = pd.read_csv('Donnees/train.csv')

        # encode chaque classe par un entier
        self.encodeur = LabelEncoder()
        df_entrainement.species = self.encodeur.fit_transform(df_entrainement.species)

        # étape de normalisation des données ?

        self.T_entrainement = np.array(df_entrainement.iloc[:, 1])
        self.X_entrainement = np.array(df_entrainement.iloc[:, 2:])

        #ajout de la normalisation des données

        if(normal):
            self.X_entrainement = StandardScaler().fit_transform(self.X_entrainement)

        df_test = pd.read_csv('Donnees/test.csv')
        self.id_test = df_test.id
        self.X_test = np.array(df_test.iloc[:, 1:])

        if (normal):
            self.X_entrainement = StandardScaler().fit_transform(self.X_entrainement)
            self.X_test = StandardScaler().fit_transform(self.X_test)
