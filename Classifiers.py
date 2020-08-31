import pickle 
import pandas as pd
import numpy as np
import re
from datetime import datetime
import time
import gzip
import statistics

# Clean
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import SnowballStemmer
from nltk import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from yapf.yapflib.yapf_api import FormatCode

# Model
import sklearn
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold 

# Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Train / Test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve
 
# Keras
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Flatten, SimpleRNN, GRU, Bidirectional
from keras.layers import Conv1D, Activation, MaxPooling1D, GlobalMaxPooling1D, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from gensim.models import KeyedVectors




def main():
    print("This module contains functions for Machine Learning Classifiers")

def logistic_regression(X_train, Y_train, X_test, Y_test):
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    pred = lr.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def Naive_Bayes(X_train, Y_train, X_test, Y_test):
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def RandomForest(X_train, Y_train, X_test, Y_test):
    clf = RandomForestClassifier(n_estimators=100
                                 , random_state=0
                                 , n_jobs=1)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def Bagging(X_train, Y_train, X_test, Y_test):
    bag = BaggingClassifier()
    bag.fit(X_train, Y_train)
    pred = bag.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def Boosting(X_train, Y_train, X_test, Y_test, max_depth=5, learning_rate=0.1, objective = 'binary:logistic', gamma=0
             , n_estimators=100, min_child_weight=3, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.1):
     
    boost = XGBClassifier(max_depth = max_depth, learning_rate = learning_rate, objective=objective, n_estimators=n_estimators
                          , min_child_weight=min_child_weight, subsample = subsample, colsample_bytree = colsample_bytree
                          , reg_alpha=reg_alpha, gamma=gamma)
    boost.fit(X_train, Y_train) 
    pred = boost.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def SVClassifier(X_train, Y_train, X_test, Y_test):
    svm = SVC()
    svm.fit(X_train, Y_train)
    pred = svm.predict(X_test)
    return accuracy_score(Y_test, pred), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def result(model, optimiser, epochs, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
    model.compile(optimizer = optimiser, loss = "binary_crossentropy", metrics = ["acc", precision_m, recall_m, f1_m])
    history = model.fit(X_train, Y_train, epochs = epochs, batch_size = 128, validation_data = (X_validation, Y_validation))
    loss, accuracy, precision, recall, f1_score = model.evaluate(X_test, Y_test)#, verbose=0)
    return loss, accuracy, precision, recall, f1_score

def rnn_types(unit_par, optimiser=optimiser, num_dense=1, dense_unit=dense_unit, num_epochs=num_epochs
               , batch_size=batch_size, activation=activation, dense_activation=dense_activation, conv_activation=False
               , emb_type=False, emb_mat=False, srnn_type=False, gru_type=False, lstm_type=False, bidir_type=False
               , return_sequences=False, X_train=X_train_RNN, X_validation=X_validation_RNN, X_test=X_test_RNN
               , num_conv=False, conv_unit=False, kernel_size=False, dropout=False, dropout_rate=False, batch_norm=False
               , num_lstm=False, num_bidir=False, lstm_unit2=False, lstm_unit3=False, bidir_unit2=False, bidir_unit3=False):
    '''This function is used to perform simple RNN, GRU, (Stacked) LSTM, (Stacked) Bidirectional LSTM models and provide 
    flexibility in choosing different layer units, optimisers, number of dense layers, dense units, number of epochs,
    batch size, activation functions, word embedding layers, dropout layers, dropout rate, batch normalisation layers, 
    flatten layers, data input, convolutional layers, filter size and kernel size.'''
    model = Sequential()
    
    # Word Embedding
    if emb_type==True:
        model.add(Embedding(len(emb_mat), embedding_dimensions, weights=[emb_mat], input_length=max_length, trainable=False))
        
    # Convolutional Layers
    if num_conv>0:
        model.add(Conv1D(filters=conv_unit, kernel_size=kernel_size, padding='same', activation=conv_activation))
        model.add(MaxPooling1D(pool_size=2))
        if dropout==True:
            model.add(Dropout(dropout_rate))
        if batch_norm==True:
            model.add(BatchNormalization()) 
    if num_conv>1:
        model.add(Conv1D(filters=int(conv_unit/2), kernel_size=kernel_size-1, padding='same', activation=conv_activation))
        model.add(MaxPooling1D(pool_size=2))
        if dropout==True:
            model.add(Dropout(dropout_rate))
        if batch_norm==True:
            model.add(BatchNormalization())
    if num_conv>2:
        model.add(Conv1D(filters=int(conv_unit/4), kernel_size=kernel_size-2, padding='same', activation=conv_activation))
        model.add(MaxPooling1D(pool_size=2))
        if dropout==True:
            model.add(Dropout(dropout_rate))
        if batch_norm==True:
            model.add(BatchNormalization())
    
    # RNN Model (Simple RNN, GRU, LSTM, Bidirectional)
    if srnn_type==True:
        model.add(SimpleRNN(units=unit_par, activation=activation, return_sequences=return_sequences))
    if gru_type==True:
        model.add(GRU(units=unit_par, activation=activation, return_sequences=return_sequences))
    if lstm_type==True:
        model.add(LSTM(units=unit_par, activation=activation, return_sequences=return_sequences))
    if bidir_type==True:
        model.add(Bidirectional(LSTM(units=unit_par, activation=activation, return_sequences=return_sequences)))
      
    # Stacked LSTM and Bidirectional
    if num_lstm>1:
        model.add(LSTM(units=lstm_unit2, activation=activation, return_sequences=return_sequences))
    if num_lstm>2:
        model.add(LSTM(units=lstm_unit3, activation=activation, return_sequences=return_sequences))                               
    if num_bidir>1:
        model.add(Bidirectional(LSTM(units=bidir_unit2, activation=activation, return_sequences=return_sequences)))
    if num_bidir>2:
        model.add(Bidirectional(LSTM(units=bidir_unit3, activation=activation, return_sequences=return_sequences)))

    # Dense Layers
    if num_dense>1:
        model.add(Dense(dense_unit, activation=dense_activation))
    if num_dense>2:
        model.add(Dense(int(dense_unit/(2**1)), activation=dense_activation))
    if num_dense>3:
        model.add(Dense(int(dense_unit/(2**2)), activation=dense_activation))
    if num_dense>4:
        model.add(Dense(int(dense_unit/(2**3)), activation=dense_activation))
    if num_dense>5:
        model.add(Dense(int(dense_unit/(2**4)), activation=dense_activation))
    if num_dense>6:
        model.add(Dense(int(dense_unit/(2**5)), activation=dense_activation))
    if num_dense>7:
        model.add(Dense(int(dense_unit/(2**6)), activation=dense_activation))
    if num_dense>8:
        model.add(Dense(int(dense_unit/(2**7)), activation=dense_activation))
                                                            
    # Return Sequences
    if return_sequences ==True:
        model.add(Flatten())
                                
    # Dense Layer for Final Classification
    model.add(Dense(1, activation="sigmoid"))
    
    # Train Model
    model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=["acc", precision_m, recall_m, f1_m])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs
                        , validation_data = (X_validation, Y_validation))


    # Test Model
    loss, accuracy, precision, recall, f1_score = model.evaluate(X_test, Y_test)
    
    # Return Test Accuracy
    return accuracy, precision, recall, f1_score
            
if __name__ == '__main__':   # Only executed if it is run as a script
    main()

