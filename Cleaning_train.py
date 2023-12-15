# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import unicodedata
import re
import nltk


nltk.download('wordnet')
nltk.download('stopwords')

# load Java Script file

#with open('practice_data.json', encoding='utf-8') as f:
with open('train.json', encoding='utf-8') as f:    
    data = json.load(f)
    
# nb of recipes
L_data = len(data)
print("Il y a %s recettes." % L_data)

# nb of cooking styles
#nb_cook = set()
#nb_cuis = {}  # dictionaire des pays
#ct = 1
#for i in range(L_data):
#    pays = data[i]["cuisine"]
#    if pays not in nb_cook:
#        nb_cuis[pays] = ct
#        nb_cook.add(pays)
#        ct += 1
#print("Il y a %s pays." % (ct-1))


### Fonctions de nettoyage du texte brut ###

def lemmatizer(liste):
    ''' .supprime tous les stopwords de liste
        .réduit les mots inclus dans la liste à leur lemme via nltk.stem WordNetLemmatizer
        .supprime des mots inutiles additionnels
    '''
    sw = stopwords.words('english')
    liste = [token for token in liste if token not in sw]
    
    lemmatizer = WordNetLemmatizer()
    liste = [lemmatizer.lemmatize(token) for token in liste]
    
    useless = []
    with open('Useless_words.txt', encoding='utf-8') as f:
        for line in f:
            useless.append(line[:-1])
    liste = [token for token in liste if token not in useless]
    
    return liste
    

def remove_parenthesis(word):
    '''suppression de (...)
    '''
    for i in range(len(word)):
        if word[i] == '(':
            d1 = i  # on stocke indice de debut
        if word[i] == ')':
            d2 = i  # on stocke indice de fin
    if d1 == 0:
        return word.split(')')[1]
    elif d2 == len(word) - 1:
        return word.split('(')[0]
        
        
def remove_brand(word, ct_m):
    ''' .supprime les caractères spéciaux R et TN 
        .concatène avec _ les noms de marque
        - word: mot à nettoyé
        - ct_m: ensemble des mots de marque mis à jour à chaque que la fonction est appelée
    '''
    # traite la cas "I can't believe it's not butter"
    if 'not butter' in word:
        return 'not_butter', ct_m

    w1 = word.replace('\N{TRADE MARK SIGN}', '\N{REGISTERED SIGN}') \
        .split('\N{REGISTERED SIGN}')

    if len(w1) == 2:

        word = "_".join(re.findall(r"\b\w\w+\b", w1[0])) + " " + w1[1]
        return word, ct_m
    elif len(w1) == 3:
        ct_m.append(w1[0] + " " + w1[1])
        word = "_".join(re.findall(r"\b\w\w+\b", w1[0])) + "_" \
               + "_".join(re.findall(r"\b\w\w+\b", w1[1])) + " " + w1[2]
        return word, ct_m
    
    
def cleaning1(word, ct_m):#, ct_m_all):
    ''' Fonction de nettoyage des mots bruts d'ingrédients:
    - met tous les caractères en minuscules
    - supprime les termes entre parenthèses
    - supprime les accents
    - supprime les chiffres
    - supprime les caractères spéciaux 
    - supprime la virgule et ce qui suit (description de cuisson)
    - suppression des caractères spéciaux de marques R et TN
    - remplacement de la marque I can't believe it's not butter par le terme not_butter
    - sauve une liste des marques
    Retourne le mot nettoyé
    '''
    # Tout mettre en minuscules
    word = word.lower()

    # suppression de (...)
    if '(' in word:
        word = remove_parenthesis(word)

    # supprimer descriptif après virgule
    if ',' in word:
        word = word.split(',')[0]

    # Repérer les noms de marques avec caractères spéciaux R et TN
    # Concaténer les nom de marques avec _
    # Simplifier la marque I can't believe it's not butter par not_butter
    # Créer un set contenant les marques (à utiliser ultérieurement)
    if '\N{REGISTERED SIGN}' in word or '\N{COPYRIGHT SIGN}' in word \
            or '\N{TRADE MARK SIGN}' in word:
        word, ct_m = remove_brand(word, ct_m)

    # supprimer les accents
    #"Mn" stands for Nonspacing_Mark
    word = ''.join(c for c in unicodedata.normalize('NFD', word)
                   if unicodedata.category(c) != 'Mn')

    # supprimer les chiffres et les caractères spéciaux restants -, &, %, !, /, ' ...
    word = ' '.join(re.findall(r"\b\w\w+\b", word))
    word = ' '.join(re.findall(r"\b\D\D+\b", word))

    return word, ct_m

def cleaning_marques(word, ct_m):
    '''
    concatène avec _ les mots de marques connus dans ct_m qui ne sont pas indiqués 
    avec un caractère spécial R ou TN.
    '''

    if 'not butter' in word:
        return "not_butter"

    for marque in ct_m:
        if marque in word:
            w1 = word.split(marque)
            m1 = "_".join(marque.split())
            if len(w1) == 2:
                return m1 + " " + w1[1]
            elif len(w1) == 3:
                return w1[0] + " " + m1 + " " + w1[2]

    return word

def cleaning2(liste):
    '''
    Fonction de nettoyage des listes d'ingrédients:
    -- supprime les stopwords
    -- reformate les mots de la liste en appliquant de la lemmalization 
    Retourne la nouvelle liste de mots
    '''
    # supprime stopwords, uselesswords et lemmatize les mots de la liste
    liste = lemmatizer(liste)
    
    ## corrections additionelles
    
    # corrige mayonnais en mayonnaise
    k = 0
    for word in liste:
        if word == "mayonnais":
            liste[k] = "mayonnaise"
            break
        k += 1
        
    # corrige culantro an cilantro
    k = 0
    for word in liste:
        if word == "culantro":
            liste[k] = "cilantro"
            break
        k += 1 
        
    #conctène des mots très courants:
    if "sesame" in liste and "oil" in liste:
        liste.remove("sesame")
        liste.remove("oil")
        liste.append("sesame_oil")
    
    if "olive" in liste and "oil" in liste:
        liste.remove("olive")
        liste.remove("oil")
        liste.append("olive_oil")  
        
    if "soy" in liste and "sauce" in liste:
        liste.remove("soy")
        liste.remove("sauce")
        liste.append("soy_sauce")
    
    if "fish" in liste and "sauce" in liste:
        liste.remove("fish")
        liste.remove("sauce")
        liste.append("fish_sauce")
        
    if "sour" in liste and "cream" in liste:
        liste.remove("sour")
        liste.remove("cream")
        liste.append("sour_cream")
        
    if "creme" in liste and "fraiche" in liste:
        liste.remove("creme")
        liste.remove("fraiche")
        liste.append("creme_fraiche") 
    
    return liste

### Construction de la Dataframe de données structurées

df_Mat_ingr_pays = pd.DataFrame(columns=['pays','ingredients'])
ct_ingr = []     # compteur des ingrédients
dico_ingr = {}      # décompte des ingrédients
for i in range(L_data):
    line = data[i]["ingredients"]
    pays = data[i]["cuisine"]
    
    ingr = []
    ct_m = []
    for val in line:
        word, ct_m = cleaning1(val, ct_m)
        word = cleaning_marques(word, ct_m)
        w1 = word.split()
        ingr.extend(w1)
        
    ingr = cleaning2(ingr)
    
    for val in ingr:
        if val not in ct_ingr:
            ct_ingr.append(val)
            dico_ingr[val] = 1
        else:
            dico_ingr[val] += 1

    df_Mat_ingr_pays.loc[i, 'ingredients'] = ingr
    df_Mat_ingr_pays.loc[i, 'pays'] = pays

# Trouver le nombre d'ingrédients de présence inférieure à 3 
# et les supprimer de la dataframe

few_ingr = []
for key, value in dico_ingr.items():
    if value < 3:
        few_ingr.append(key)
for x in few_ingr:
    ct_ingr.remove(x)

for i in range(L_data):
    ingr = df_Mat_ingr_pays.loc[i, 'ingredients']
    for x in few_ingr:
        while x in ingr:
            ingr.remove(x)
    ingr = " ".join(ingr)
    df_Mat_ingr_pays.loc[i, 'ingredients'] = ingr

print('Apres nettoyage, il y a %s ingredients.' %len(ct_ingr))


with open('liste_ingr_text.txt', 'w') as f:
    for item in ct_ingr:
        f.write("%s\n" % item)
liste_ingr = np.asarray(ct_ingr)
np.savetxt('liste_ingr_text.csv', liste_ingr, delimiter=',', fmt='%d')
df_Mat_ingr_pays.to_csv('Mat_ingr_pays.csv', sep=',', encoding='utf-8', index=None, header=True)
