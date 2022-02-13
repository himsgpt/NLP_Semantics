# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:59:33 2019

@author: higupta

"""

import Semantic_Search.src.global_var as g_var
import pandas as pd
import logging
import sys
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from Semantic_Search.src.EDA import cleanData
from datetime import datetime
from gensim.models.phrases import Phrases, Phraser
import string

class Semantics:
    '''
    Semantic pipeline blueprint
    '''

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=g_var.LOG_FILE, filemode='w', 
                        format='%(name)s - %(levelname)s - %(message)s', level = logging.DEBUG)
        
    def __init__(self):
        self.logger.info("started")
        
# =============================================================================
# Pretrained Model
# =============================================================================
    def pretrained(self):
        """
        Google pretrained models
        """
        self.logger.info("Google pretrained models")
        # =============================================================================
        #Word2Vec 
        # =============================================================================
        try:
            model = KeyedVectors.load_word2vec_format(g_var.WORD_TO_VEC_FILE_PATH, binary=True)
        except Exception as e:
            self.logger.critical("Unable to Load file" + str(e))
            sys.exit(0)
        # Example
            result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
            print(result)
            print(model.most_similar('cloud'))
        
        # =============================================================================
        # Glove
        # =============================================================================        
        #converting the glove file to vector format
        self.logger.info("converting the glove file to vector format")
        glove_input_file = g_var.GLOVE_FILE_PATH
        word2vec_output_file = 'C:\\Users\\HIGUPTA\\Downloads\\Data_Science\\Projects\\Semantic_Search\\data\\in\\glove.6B.300d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)
        
        #Model after picking up
        filename = g_var.FILENAME
        try:
            model2 = KeyedVectors.load_word2vec_format(filename, binary=False)
        except Exception as e:
            self.logger.critical("Unable to Load file" + str(e))
            sys.exit(0)
            
        # Testing the results
        print(model2.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
        print(model2.similarity('woman', 'man'))
        print(model2.most_similar("cloud"))
        print(model2["woman"])  #vector for word in 300 dimensional
        'Not good result from above models, retrain your model'
    
# =============================================================================
# Prepare the data
# =============================================================================
    def gensim_list(self):
        '''
        using gensim for training models on your prepared data
        '''
        self.logger.info("using gensim for training models on your prepared data")
        
        # =============================================================================
        # Cleaning the data
        # =============================================================================
        data = pd.read_csv(g_var.DATALAKE, encoding="ISO-8859-1")
        data.head(5)
        data.dtypes
        start_time = datetime.now()
        data['Quotes_cleaned'] = data["quote"].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, word_len=True, stemming=False, lemmatize= False))
        len(data)
        data["Quotes_cleaned"].iloc[56001]
        
        #Removing low freq words from the list        
        self.logger.info("Removing low freq words from the list") 
        freq = pd.Series(' '.join(data['Quotes_cleaned']).split()).value_counts()[-10000:]
        print(freq)
        freq = list(freq.index)
        data['Quotes_cleaned'] = data['Quotes_cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        (data["Quotes_cleaned"].iloc[20])
        
        ##Removing high freq words from the list
        self.logger.info("Removing high freq words from the list")
        freq = pd.Series(' '.join(data['Quotes_cleaned']).split()).value_counts()[:10]
        print(freq)
        freq = list(freq.index)
        data['Quotes_cleaned'] = data['Quotes_cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        data["Quotes_cleaned"].iloc[657]
        
        ## remove punctuations
        exclude = set(string.punctuation)
        data['Quotes_cleaned'] = data['Quotes_cleaned'].apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in exclude))
        data["Quotes_cleaned"].iloc[56001]
        
        ## remove numbers
        
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        # =============================================================================
        # Tokenization and training
        # =============================================================================
        documents = data['Quotes_cleaned'].apply(word_tokenize)
        documents[56001]
        
        # =============================================================================
        # bigram phrases 
        # =============================================================================
        self.logger.info("bigram phrases")
        start_time = datetime.now()
        bigram = Phrases(documents, min_count=5, threshold=1)
        bigram_phraser = Phraser(bigram)
        bigram_docs = documents.apply(lambda x: (bigram_phraser[x]))
        bigram_docs[56001]
        
        # =============================================================================
        # Trigram phrases 
        # =============================================================================
        
        # =============================================================================
        # build vocabulary and train model
        # =============================================================================
        self.logger.info("build vocabulary and train model")
        model3 = gensim.models.Word2Vec(bigram_docs,size=300,
                 window=10,min_count=5,workers=10, sg=1)
        model3.train(documents, total_examples=len(bigram_docs), epochs=50)
        end_time = datetime.now()
        self.logger.info('Duration: {}'.format(end_time - start_time))
        
        model3.wv.save_word2vec_format(g_var.OUTPUT)
        #model3.save("gartner_word2vec.model")
            
# =============================================================================
# Check relative ranked similarity
# =============================================================================
    def check_results(self,x):
        """
        testing the results and comparing with google word2vec
        Semantics are more similar as trained on gartners text
        """
        self.logger.info("testing the results and comparing with google word2vec")
        global model2
        global model3
        print("Gartners result")
        try:
            a= (model3.wv.most_similar('income', topn=10))
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
    gensim_obj = Semantics()
    gensim_obj.gensim_list()
    

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