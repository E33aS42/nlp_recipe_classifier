# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

mat_ingr = pd.read_csv('Mat_ingr_pays_lem_2_usel_concat.csv')

L_data = len(mat_ingr)
list_ingr = []
ID_Pays = {}
nb_pays = set()
ct = 1
liste_pays = []

# Numérise les pays donnant un numéro de classe de 1 à 20
for i in range(L_data):
    P = mat_ingr.loc[i, 'pays']

    if P not in nb_pays:
        liste_pays.append(P)
        nb_pays.add(P)
        ID_Pays[P] = ct
        ct += 1
        
for i in range(L_data):
    mat_ingr.loc[i, 'pays'] = ID_Pays[mat_ingr.loc[i, 'pays']]  

# Séparation des données train-test pour utilisation ultérieure
    
train, test = train_test_split(mat_ingr, test_size=0.1)

test.to_csv('mat_ingr_test.csv', sep=',', encoding='utf-8', index=None, header=True)

# Binarisation des données

X = train.loc[:, 'ingredients']
Y = train.loc[:, 'pays']

vect = CountVectorizer()  # Module qui convertie une collection de textes bruts en un matrice de comptes termes-ducuments
vect.fit(X)               # Apprends un vocabulaire de l'ensemble des tokens (mots contenus dans l'ensemble des textes.
Xtr = vect.transform(X)   # Binarisation des données
CV_save = "CountVect_dict.sav"
joblib.dump(vect.vocabulary_, CV_save)    # Enregistre le vocabulaire de CountVectorizer
trans = TfidfTransformer()      # Transforme une matrice de comptes en une normalisation (représentation) tf-idf
# we fit and transform data from count matrices previously obtained in Unigram form
Xtr_tfidf = trans.fit_transform(Xtr)

X_tr_dense = Xtr.todense()
X_tr_tfidf_dense = Xtr_tfidf.todense()
sparse.save_npz("X.npz", Xtr)
Y.to_csv('Y.csv', sep=',', encoding='utf-8', index=None, header=True)
sparse.save_npz("X_tfidf.npz", Xtr_tfidf)
np.savetxt('X_dense.csv', X_tr_dense, delimiter=',', fmt='%d')
np.savetxt('X_tfidf_dense.csv', X_tr_tfidf_dense, delimiter=',', fmt='%d')