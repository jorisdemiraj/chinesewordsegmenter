
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


train_path="../resources/path.xml"
#dev_path="../resources/path2.xml"


TInput, TLabel, Fullline, Fulllabelline, lens, senlen=preprocess(train_path)
X_trainu,X_trainb,Y_train,vocab,vocabu=buildvector(TInput, TLabel, Fullline, Fulllabelline)

#DInput, DLabel, DevFullline, DevFulllabelline,lendev,senlendev=preprocess(dev_path)
#X_devu, X_devb, Y_dev,vocabdev,vocabudev =buildvector(DInput, DLabel, DevFullline, DevFulllabelline, vocab, vocabu)

#INITIAL HYPER PARAMETERS
VOCAB_SIZE1 = (len(vocab)+1)
VOCAB_SIZE2=(len(vocabu)+1)
HIDDEN_SIZE = 256

#This is the model's architecture

def create_keras_model(vocab_size1,vocab_size2,  hidden_size):
    print("Creating KERAS model")
    
  

    
    model = K.models.Sequential()
   

    un=K.layers.Input(shape=(None,),name='unigrams')
    em_unigram = K.layers.Embedding(vocab_size2, 64, mask_zero=True)(un)
              
    
    bi=K.layers.Input(shape=(None,), name='bigrams')
    em_bigram = K.layers.Embedding(vocab_size1, 32, mask_zero=True)(bi)


    merged= K.layers.concatenate([em_unigram, em_bigram], axis=-1)
    
    dropped = K.layers.Dropout(0.3)(merged)
    lstm_result = K.layers.Bidirectional(K.layers.LSTM(hidden_size, dropout=0.3, recurrent_dropout=0.25, return_sequences=True))(dropped)
 
    output  = K.layers.TimeDistributed(K.layers.Dense(4, 'softmax'))(lstm_result)
    
    model = K.models.Model(inputs=[un, bi], outputs=output)
    optimizerA = K.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=optimizerA, metrics=['acc'])

    return model
    



batch_size = 128
epochs = 10

model = create_keras_model(VOCAB_SIZE1,VOCAB_SIZE2,  HIDDEN_SIZE)
model.summary()

#build a early stopping option to prevent any overfitting or increase of loss

cbk = K.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
print("\nStarting training...")
#model.fit([X_trainu,X_trainb], Y_train, epochs=epochs, batch_size=batch_size,
#         shuffle=True,validation_data=([X_devu,X_devb],Y_dev), callbacks=[cbk]) 
print("Training complete.\n")

#save the model configuration and weights
#model.save('my_model.h5')

