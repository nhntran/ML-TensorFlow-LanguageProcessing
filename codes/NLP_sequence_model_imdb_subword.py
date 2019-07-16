
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
#/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/
#MachineLearning/TensorFlow_LanguageProcessing/codes/

# Need to use tensorflow 2.0.0
# Check version with 'print(tf.__version__)'
#'pip install tensorflow==2.0.0-beta0' for install
# if it's not 2.00 => include this line of code:
# 'tf.enable_eager_execution()'

# imdb dataset is in the tensorflow-datasets
# install the tensorflow dataset:
# 'pip install -q tensorflow-datasets'

from __future__ import absolute_import, division, print_function,\
                        unicode_literals
# import os
# import io
import tensorflow as tf
import tensorflow_datasets as tfds
# import numpy as np                            
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def imdb_data_import(buffer_size, batch_size):
    #load the subwords8k data
    imdb, info = tfds.load("imdb_reviews/subwords8k", 
                             with_info = True, as_supervised = True)
    train_data, test_data = imdb['train'], imdb['test']

    tokenizer = info.features['text'].encoder
    train_data = train_data.shuffle(buffer_size)
    #buffer_size: A tf.int64 scalar tf.Tensor, representing 
    #the number of elements from this dataset 
    #from which the new dataset will sample.
    train_data = train_data.padded_batch(batch_size, train_data.output_shapes)
    test_data = test_data.padded_batch(batch_size, test_data.output_shapes)

    return tokenizer, train_data, test_data

def buiding_nn_model_lstm_single(corpus_size, embedding_dim,
                            train_data, test_data, num_epochs):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        # output shape with 64 unit: (None, 128) 
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_data, epochs = num_epochs,
                                validation_data = test_data)
    #after training, saving the model into the .h5 file
    model.save('TF_NLP_sequence_model_single.h5') 

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #evaluate the model
    model_evaluation(acc, val_acc, loss, val_loss)

    e = model.layers[0]
    weights = e.get_weights()[0]

    return weights

def buiding_nn_model_lstm_multi(corpus_size, embedding_dim,
                            train_data, test_data, num_epochs):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
                                        return_sequences = True)),
        # return_sequences = True: instruct an LSTM to feed the 
        #next LSTM in the sequence When stacking LSTMs
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_data, epochs = num_epochs,
                                validation_data = test_data)
    #after training, saving the model into the .h5 file
    model.save('TF_NLP_sequence_model_multi.h5') 

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #evaluate the model
    model_evaluation(acc, val_acc, loss, val_loss)

    e = model.layers[0]
    weights = e.get_weights()[0]

    return weights

def buiding_nn_model_1D_Convolutional(corpus_size, embedding_dim,
                            train_data, test_data, num_epochs):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim),
        tf.keras.layers.Conv1D(128, 5, activation = 'relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_data, epochs = num_epochs,
                                validation_data = test_data)
    #after training, saving the model into the .h5 file
    model.save('TF_NLP_sequence_model_1Dconv.h5') 

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #evaluate the model
    model_evaluation(acc, val_acc, loss, val_loss)

    e = model.layers[0]
    weights = e.get_weights()[0]

    return weights

def writing_v_m_files(weights, tokenizer):
    vfile = open('data/imdb-sub/vecs.tsv','w', encoding ='utf-8')
    mfile = open('data/imdb-sub/meta.tsv','w', encoding ='utf-8')

    for num in range(1, tokenizer.vocab_size):
        word = tokenizer.decode([num])
        embeddings = weights[num]
        mfile.write(word+'\n')
        vfile.write('\t'.join([str(x) for x in embeddings])+'\n')
    vfile.close()
    mfile.close()

def model_evaluation(acc, val_acc, loss, val_loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc, 'r', "Training Accuracy")
    plt.plot(epochs,val_acc, 'b')
    plt.title("Training and validation accuracy")
    plt.xlabel ("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation_Accuracy"])
    plt.show()

    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("Training and validation loss")
    plt.xlabel ("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation_Loss"])
    plt.show()

########################################################################
# The main() function
def main():
    
    ### changable parameters:
    buffer_size = 10000
    batch_size = 64
    # dimensions for the vector representing the subwords encoding
    embedding_dim = 64
    num_epochs = 1 #number of epochs for training

    #import data and generate tokenizer
    tokenizer, train_data, test_data = imdb_data_import(buffer_size,
                                                            batch_size)
    corpus_size = tokenizer.vocab_size
    
    #building model 1 - Single layer LSTM
    #weights = buiding_nn_model_lstm_single(corpus_size, embedding_dim,
    #    train_data, test_data, num_epochs)

    #building model 2 - Multi layer LSTM
    # weights = buiding_nn_model_lstm_multi(corpus_size, embedding_dim,
    #     train_data, test_data, num_epochs)

    weights = buiding_nn_model_1D_Convolutional(corpus_size, embedding_dim,
                           train_data, test_data, num_epochs)

    # Second time running: Loading the model again
    # new_model = tf.keras.models.load_model('TF_NLP_.h5')
    # e = new_model.layers[0]
    # weights = e.get_weights()[0]

    #writing_v_m_files(weights, tokenizer)

    # Viewing the words and their vectors with Embedding Projector 
    # https://projector.tensorflow.org/
#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

