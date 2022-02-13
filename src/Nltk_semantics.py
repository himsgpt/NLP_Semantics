# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:26:42 2019

@author: higupta
"""

# =============================================================================
# get synonyms use nltk
# =============================================================================
from nltk.corpus import wordnet
syn = wordnet.synsets("pain")
print(syn)
# get a list of definitions for those words
for i in range(len(syn)):
    print(syn[i].definition()) 
#    print(syn[0].examples())
    
#hypernyms are the synsets that are more general
#hyponyms are the synsets that are more specific

synonyms = []
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

# =============================================================================
# Semantic similarity
# =============================================================================
#1 / (shortest_path_distance(synset1, synset2) + 1)

syn1 = wordnet.synsets("tiger")
print(syn1)
syn2 = wordnet.synsets("cat")
syn3 = wordnet.synsets("dodo")
syn1[0].path_similarity(syn2[0])
syn1[0].path_similarity(syn3[0])

# =============================================================================
# Data Cleaning
# =============================================================================
import Semantic_Search.src.global_var as g_var
import pandas as pd
from Semantic_Search.src.EDA import cleanData

quotes = pd.read_csv(g_var.PROPER_QUOTE, encoding="ISO-8859-1")
Quotes_cleaned = quotes['quote'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=False, lemmatize= True))
Quotes_cleaned.iloc[20]

# =============================================================================
# TF-idf distribution
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
#min document frequency 5 and max document frequcny is 80%
vector= TfidfVectorizer(ngram_range=[1,2], min_df=10, max_df=0.8)
X = vector.fit_transform(Quotes_cleaned)
X.shape

# =============================================================================
# LSA getting latent factors
# =============================================================================
from sklearn.decomposition import TruncatedSVD
import numpy as np
# SVD represent documents and terms in vectors 
#check hypetrparameters
svd_model = TruncatedSVD(n_components=15, algorithm='randomized', n_iter=10, random_state=120)
svd_model.fit(X)

# check amount of variance explained by these topics
svd_model.components_
var = (svd_model.explained_variance_ratio_)
print(-np.sort(-var))
print(var.sum())

import matplotlib.pyplot as plt
plt.plot(-np.sort(-var))
plt.show

# =============================================================================
# Get features ranked as per variance explained by them
# =============================================================================
terms = vector.get_feature_names()
topics = []
no = []
# Top 50 terms for every topic
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:1]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")
        topics.append(t[0])
        no.append(str(i))
        
topic_keyword = pd.DataFrame({"Topic_no":no, "Keywords": topics})  
topic_keyword.to_csv('C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\out\\topics.csv', index =False, sep = ',')      
#np.savetxt('C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\out\\topics.csv',
#           topics, delimiter=",", fmt='%s')