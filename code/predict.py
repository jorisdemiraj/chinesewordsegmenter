from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from typing import Tuple, List, Dict
#get_ipython().system('pip install keras')
#get_ipython().system('pip install ipynb')
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Uncomment this to run on Mac OS
import matplotlib.pyplot as plt
import re
import collections
from random import shuffle
from keras.models import load_model
from Preprocess import preprocess, buildvector,savegoldtofile
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow.keras as K
from nltk.util import ngrams
import pickle
from Model import create_keras_model
from numpy import argmax



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):

   
    #The prediction works by uploading first of all the vocabularies used in training phase
    vocab=dict() 
    vocabu=dict()
    with open('../resources/vocab.pkl', 'rb') as f:
        vocab= pickle.load(f)
    with open('../resources/vocabu.pkl', 'rb') as f:
        vocabu= pickle.load(f)
    
    #we pass the whole test dataset through the preprocessing phase and save the gold data in a file (this in case the input file has whitespaces, in which case a gold data set can be retrieved, otherwise
    # it will just generate a random variable file that we can just ignore)
        
    TInput, TLabel, TFullline, TFulllabelline, lens,senlens=preprocess(input_path)
    X_testu, X_testb, Y_test,vocabt,vocabut =buildvector(TInput, TLabel, TFullline, TFulllabelline, vocab, vocabu, lens)
    savegoldtofile(TLabel)

    #initialize the model and upload the weights and configuration from the file
    model = create_keras_model((len(vocab)+1),(len(vocabu)+1), 256)
    model=load_model(resources_path)

    #proceed with the prediction. We feed the X vectors to the predict function and get back a vector with one hot encoding. We reverese the encoding through the argmax function, retrieve the data in numerical form
    #and proceed by assigning the label corresponding the the value
    #in the end we remove the padding that we added through the preprocessing and save the file 

    prediction = model.predict([X_testu, X_testb])
    text_file = open(output_path, "w+")
    sen=[]
    count=0
    for row in prediction:
       
        line=[]
        for element in row:
            val=np.argmax(element)
            if val==0:
                line.extend("B")
            elif val==1:
                line.extend("E")
            elif val==2:
                line.extend("I")
            else:
                line.extend("S")
        linez=''.join(line)
        linez2=linez[:senlens[count]]
        sen.append(linez2)
        text_file.write(linez2+'\n')
        count+=1
    
    text_file.close()
    
if __name__ == '__main__':
    args = parse_args( )
    predict(args.input_path, args.output_path, args.resources_path)
