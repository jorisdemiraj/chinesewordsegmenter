#Libraries used for this Homework

from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from typing import Tuple, List, Dict
#get_ipython().system('pip install keras')
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Uncomment this to run on Mac OS
import matplotlib.pyplot as plt
import re
import collections
from random import shuffle
from keras.models import load_model
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow.keras as K
from nltk.util import ngrams
import pickle

#This function is used to extract the data from the dataset and append it as sentence by sentence in a list or as word by word. In both ocassions , actions are taken to make sure
#each sentence is stripped off whitespaces.

def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    
    words = []
    sentences=[]
    
    with open(path, encoding='utf8') as f:
        for line in f:
            
            #actions performed to remove whitespaces

            wordList= []
            linz=line[:len(line)-1].replace(" ","")
            linz2=linz.replace("\t","")
            linz3=linz2.replace("\u3000","")
            sentences.append(linz3)
            if line:
                wordList = line.split()
                words.extend(wordList)
    return words, sentences


#function used mainly to convert each word in its BIES counterpart. It takes each character, checks if its Single (S), beginning of a word (B), middle (I) or end (E)
#and it appends its label

def set_label(words) -> Tuple[List[str], List[str]]:
    
    label= []
    inputs= []
    #print(words)
    for word in words:
        if len(word)>1:
            
            inputs.append(word[0])
            label.append('B')
        
            for k in word[1:len(word)-1]:
                inputs.append(k)
                label.append('I')
            inputs.append(word[-1])
            label.append('E')
        else:
            inputs.append(word)
            label.append('S')
    return inputs, label


#function used mainly to convert each sentence in its BIES counterpart. It takes each character, checks if its Single (S), beginning of a word (B), middle (I) or end (E)
#and it appends its label

def labelintosen(sentencez: List[str], labelz: List[str]):
    label=[]
    v=0
    for sentence in sentencez:
        a=[]  
        for x in sentence:
            a.append(labelz[v])
            v+=1
        a=''.join(a)
        label.append(a)
    return label


#this function is self explanatory. It takes the sentences as an input and it splits them into bigrams

def split_into_bigrams(sentence: str) -> List[str]:
    bigrams = []
    for i in range(len(sentence)-1):
        bigram = sentence[i:i+2]
        bigrams.append(bigram)
    return bigrams

#function similar to the one above. It takes the sentences as an input and it splits them into unigrams

def split_into_unigrams(sentence: str) -> List[str]:
    bigrams = []
    for i in range(len(sentence)-1):
        bigram = sentence[i]
        bigrams.append(bigram)
    return bigrams

#this function is used to build the vocabulary based on either the bigrams or unigrams

def make_vocab(grams) -> Dict[str, int]:
    vocab = {"UNK": 0}
    
    for gram in grams:
        if gram not in vocab:
            vocab[gram] = len(vocab)
    return vocab           

#The 2 main functions of the Preprocessing file. These functions take the sentences as an input, a dictionary , either bigram or unigram dictionary and a maxlength which is the maximum length
#of the biggest sentence in our dataset. What this does is assign this length to the padding size argument which will pad each sentence to this ammount. By default its set to None while padding at 30.#
#In case of prediction, a length is specified to make sure the whole sentence gets predicted and not risk part of sentences trunkated. Both functions are similar with small changes to make sure they work with 
#either bigrams or unigrams.

def feature_vector_bigram(text,vocabulary,maxlength=None):
    feature_vector  = []
    for i in range(len(text)):
        vector =[]
        line = text[i]

        for i in range(len(line)-1):      
            bigram = line[i:i+2]

            if bigram  not in vocabulary.keys():
                vector.append(vocabulary['UNK'])
                
            else:       
                vector.append(vocabulary[bigram])
        feature_vector.append(vector)
    if maxlength==None:
        feature_vector= keras.preprocessing.sequence.pad_sequences(feature_vector,truncating='pre',padding='post',maxlen=30 )
    else:
        feature_vector= keras.preprocessing.sequence.pad_sequences(feature_vector,truncating='pre',padding='post',maxlen=maxlength )
    return feature_vector



def feature_vector_unigram(text,vocabulary,maxlength=None):
    feature_vector  = []
    for i in range(len(text)):
        vector =[]
        line = text[i]

        for i in range(len(line)):      
            unigram = line[i]     
           
            if unigram  not in vocabulary.keys():
                vector.append(vocabulary['UNK'])
            else:
                vector.append(vocabulary[unigram])
        feature_vector.append(vector)
    if maxlength==None:
        feature_vector= keras.preprocessing.sequence.pad_sequences(feature_vector,truncating='pre',padding='post',maxlen=30 )
    else:
        feature_vector= keras.preprocessing.sequence.pad_sequences(feature_vector,truncating='pre',padding='post',maxlen=maxlength )
    return feature_vector

#This function generates the Y vector of labels from the sentences we feed to it. Basically this is the numerical counterpart of the BIES form, in vector form and ready to be fed to the network

def generateLabels(sentences):
    output = {"B":0,"E":1,"I":2,"S":3}
    y= []
    text_file = open("../resources/y_train.xml", "w+")
    for word in sentences:
        label = []
        for ch  in word:
            if output.get(ch) != None:
                label.append(output.get(ch))
        
        text_file.write(str(label))
        
        y.append(label)
        
    y = K.preprocessing.sequence.pad_sequences(y,truncating='pre',padding='post',maxlen=30)
    text_file.close()
    return y



#saves to file a string of chinese characters and its bies counterpart

def savetofile(a1,b1):
    a=a1
    b=b1
    text_file = open("../resources/Inputs.xml", "wb")
    text_file.write(a.encode('utf8'))
    text_file.close()
    text_file = open("../resources/Labels.xml", "wb")
    text_file.write(b.encode('utf8'))
    text_file.close()

#saves to file both vocabularies

def savetofile2(a1,b1):
    a=a1
    b=b1
    
    with open('../resources/vocab.pkl', 'wb') as f:
        pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
    
    with open('../resources/vocabu.pkl', 'wb') as f:
        pickle.dump(b, f, pickle.HIGHEST_PROTOCOL)

#function to get the length of the biggest sentence in the dataset 

def getmaxlen(sentences):
    val=0
    lengths=[]
    for line in sentences:
        lengths.append(len(line))
        if val<len(line):
            val=len(line)
        
    return val, lengths

#the first function to be called for the preprocessing of the data. This takes care of converting the original data to a list of sentences without spaces and a list of sentences in BIES form. It does so 
#by calling the previews mentioned functions
def preprocess(path):
    words, sentences = load_dataset(path)
    inputs, label=set_label(words)
    a= ''.join(inputs)
    b= ''.join(label)
    labelsentence= labelintosen(sentences,b)
    lens,senlen=getmaxlen(sentences)
    return sentences, labelsentence,a,b, lens,senlen

#second function to be called.  This functions take the outputs of the above function as well as pre defined vocabularies if there are any and the length of the biggest sentence to aid into padding , always
#if there is any.
    
def buildvector(sentences,labelsentence,a,b,vocab=None, vocabu=None,lens=None):
    bigrams= split_into_bigrams(a)
    unigrams=split_into_unigrams(a)
    flag=0
    if vocab==None:
        flag=1
        vocab=make_vocab(bigrams)
    if vocabu==None:
        flag=1
        vocabu=make_vocab(unigrams)
    
    if lens==None:
        lens=30
    X_trainb=feature_vector_bigram(sentences,vocab,lens)
    X_trainu=feature_vector_unigram(sentences,vocabu,lens)
    Y_train=generateLabels(labelsentence)
    Y_train = K.utils.to_categorical(Y_train)
    
    savetofile(a,b)
    savetofile2(vocab,vocabu)
    return X_trainu, X_trainb, Y_train,vocab,vocabu
    
#this function is used to save the bies gold file into a file

def savegoldtofile(gold):
        sengold=[]
        text_file = open("../resources/gold.txt", "w+")
        for row in gold:
            
            text_file.write(row+'\n')
        text_file.close()
        
#the below code makes sure to call preprocessing file from the command line by feeding arguments into it at any moment without needing to run the model or predict files.
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args( )
    vocab=dict() 
    vocabu=dict()
    with open('../resources/vocab.pkl', 'rb') as f:
        vocab= pickle.load(f)
    with open('../resources/vocabu.pkl', 'rb') as f:
        vocabu= pickle.load(f)
        
    DInput, DLabel, DevFullline, DevFulllabelline, lenz, zenlen=preprocess(args.input_path)
    X_devu, X_devb, Y_dev,vocabdev,vocabudev=buildvector(DInput, DLabel, DevFullline, DevFulllabelline, vocab, vocabu, lenz)
    savegoldtofile(DLabel)




