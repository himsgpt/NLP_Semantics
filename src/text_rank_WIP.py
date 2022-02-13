# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:11:38 2019

@author: higupta
"""

import Semantic_Search.src.global_var as g_var
import pandas as pd
from Semantic_Search.src.EDA import cleanData
import numpy as np
#nltk.download('punkt') # one time execution

# =============================================================================
# Data Cleaning/Reading
# =============================================================================
df = pd.read_csv(g_var.QUOTE_PATH, encoding="ISO-8859-1")
df_cleaned = df['quote'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, word_len=True, stemming=False, lemmatize= True))
df_cleaned.iloc[20]

# =============================================================================
# Vector Representation
# =============================================================================
# Extract word vectors
word_embeddings = {}
f = open(g_var.GLOVE_FILE_PATH, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

#Vector for our sentences
sentence_vectors = []
for i in df_cleaned:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((300,))
  sentence_vectors.append(v)

# =============================================================================
# Similarity Matrix Representation
# =============================================================================
# similarity matrix
from datetime import datetime
start_time = datetime.now()
sim_mat = np.zeros([len(df), len(df)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(df)):
  for j in range(len(df)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
      
# =============================================================================
# Text Rank
# =============================================================================
import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
nx.draw(nx_graph)
scores = nx.pagerank(nx_graph)