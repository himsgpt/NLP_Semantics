# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:42:02 2019

@author: higupta
"""


# =============================================================================
# Data Cleaning
# =============================================================================
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re


def cleanData(text, lowercase = False, remove_stops = False, word_len = False, stemming = False, lemmatize=False):
    """
    Pre-Process the data for several use
    """
    stops = set(stopwords.words("english"))
    
    txt = str(text)
    #txt = re.sub(r'[^A-Za-z\s]',r'',txt)
    txt = re.sub(r'\d+',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if word_len:
        txt = " ".join([w for w in txt.split() if len(w)>2])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    
    if lemmatize:
        lt = WordNetLemmatizer()
        txt = " ".join([lt.lemmatize(w) for w in txt.split()])

    return txt


"""
import Semantic_Search.src.global_var as g_var
data = pd.read_excel(g_var.QUOTE_PATH, encoding="ISO-8859-1")
data.head(5)
data.dtypes
data['Quotes_cleaned'] = data["quote"].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=False))
len(data)
(data["Quotes_cleaned"].iloc[20])

# =============================================================================
# Distribution of words
# =============================================================================
def Distribution(x):    
     
    vec = CountVectorizer(stop_words='english', ngram_range=(1,1)).fit(data['Quotes_cleaned'])
    bag_of_words = vec.transform(data['Quotes_cleaned'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    sorted(words_freq, key = lambda x: x[1], reverse=True)[0:20]

      
# =============================================================================
# 
# =============================================================================

 """