# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:35:07 2019

@author: higupta
"""

import pandas as pd
import xlwings as xw
print("check1")
# =============================================================================
# Gartner Results
# =============================================================================
from gensim.models import KeyedVectors
path = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\src\\Gensim_VBA.xlsm'
filename = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\out\\gartner_word2vec2'
model1 = KeyedVectors.load_word2vec_format(filename, binary=False)

#filename = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\in\\glove.6B.50d.txt.word2vec'
#model2 = KeyedVectors.load_word2vec_format(filename, binary=False)

def check():
    word = pd.read_excel(path)
    try:
        a = (model1.wv.most_similar(word['Enter_Keyword'][0], topn=10))
        ans = pd.DataFrame(a, columns=["Similar_Keywords", "Similarity_Score"])
        print("check2")
        wb = xw.Book.caller()
        wb.sheets[0].range("C2:E12").value = ans  
    except:
        ans = "Not found"
        wb = xw.Book.caller()
        wb.sheets[0].range("C2:E12").clear_contents() 
        wb.sheets[0].range("D2").value = ans  
        
        
# =============================================================================
# Pretrained results
# =============================================================================
"""
filename = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\in\\glove.6B.300d.txt.word2vec'
model2 = KeyedVectors.load_word2vec_format(filename, binary=False)
b = (model2.wv.most_similar('.net', topn=10))
base = pd.DataFrame(b, columns=["Similar_Keywords", "Similarity_Score"])
print(base)
print("check3")
"""
# =============================================================================
# xlwings integration
# =============================================================================

#wb.sheets[0].range("G2:I12").value = base

#xw.__path__