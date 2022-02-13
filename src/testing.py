# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:59:33 2019

@author: higupta

"""

import Semantic_Search.src.global_var as g_var
import pandas as pd

def pretrained():
    """
    Google pretrained models
    """
    # =============================================================================
    #Word2Vec model 
    # =============================================================================
    
    from gensim.models import KeyedVectors
    import os
    os.getcwd()
    model = KeyedVectors.load_word2vec_format(g_var.WORD_TO_VEC_FILE_PATH, binary=True)
    # Example
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(result)
    print(model.most_similar('digital'))
    
    # =============================================================================
    # Glove model
    # =============================================================================
    
    #converting the glove file to vector format
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_input_file = g_var.GLOVE_FILE_PATH
    word2vec_output_file = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\in\\glove.6B.300d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    
    #Model after picking up
    filename = '..\\Projects\\Semantic_Search\\data\\in\\glove.6B.300d.txt.word2vec'
    model2 = KeyedVectors.load_word2vec_format(filename, binary=False)
    
    # Testing the results
    print(model2.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
    print(model2.similarity('woman', 'man'))
    print(model2.most_similar("date"))
    print(model2["woman"])  #vector for word in 300 dimensional
    
    'Not good result from above models, retrain your model'

# =============================================================================
# Prepare the data
# =============================================================================
def gensim():
    
    data = pd.read_csv(g_var.DATALAKE, encoding="ISO-8859-1")
    data.head(5)
    # =============================================================================
    # Cleaning the data
    # =============================================================================
    from Semantic_Search.src.EDA import cleanData
    data['Quotes_cleaned'] = data["quote"].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, word_len=True, stemming=False, lemmatize= False))
    len(data)
    (data["Quotes_cleaned"].iloc[20])
    
    from datetime import datetime
    start_time = datetime.now()
    
    #Removing low freq words from the list 
    freq = pd.Series(' '.join(data['Quotes_cleaned']).split()).value_counts()[-10000:]
    print(freq)
    freq = list(freq.index)
    data['Quotes_cleaned'] = data['Quotes_cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    (data["Quotes_cleaned"].iloc[0])
    
    ##Removing high freq words from the list 
    '''freq = pd.Series(' '.join(data['Quotes_cleaned']).split()).value_counts()[:10]
    print(freq)
    freq = list(freq.index)
    data['Quotes_cleaned'] = data['Quotes_cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    (data["Quotes_cleaned"].iloc[0])
    '''
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    
    # =============================================================================
    # Tokenization and training
    # =============================================================================
    from nltk.tokenize import word_tokenize
    import gensim
    documents = data['Quotes_cleaned'].apply(word_tokenize)
    documents[0]
    
    # =============================================================================
    # phrases 
    # =============================================================================
    from gensim.models.phrases import Phrases, Phraser
    bigram = Phrases(documents, min_count=5, threshold=1)
    bigram_phraser = Phraser(bigram)
    bigram_docs = documents.apply(lambda x: (bigram_phraser[x]))
    bigram_docs[0]
    
    # =============================================================================
    # train
    # =============================================================================
    start_time = datetime.now()
    #build vocabulary and train model
    model3 = gensim.models.Word2Vec(bigram_docs,size=300,
             window=10,min_count=10,workers=10, sg=1)
    model3.train(bigram_docs, total_examples=len(documents), epochs=50)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    
#    model3.wv.save_word2vec_format("gartner_word2vec")
#    model3.save("gartner_word2vec.model")
    
    
    # =============================================================================
    # Check relative ranked similarity
    # =============================================================================
def main(x):
    """
    testing the results and comparing with google word2vec
    Semantics are more similar as trained on gartners text
    """
    global model2
    global model3
    print("Gartners result")
    try:
        a= (model3.wv.most_similar(x, topn=10))
        df = pd.DataFrame(a)
        print(df)
    except:
        print("not found")
            
    print('\n'+'Google result')
    try:
        b = (model2.wv.most_similar(x, topn=10))
        df = pd.DataFrame(b)
        print(df)
    except:
        print("not found")           
  
  
if __name__ == "__main__":
    main("cloud")

    
"""
# =============================================================================
# LSI
# =============================================================================
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]  # vectorize input copus in BoW format


# =============================================================================
#LDA 
# =============================================================================
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)

"""