
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

# sarcasm dataset
# Download data
# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
#     -O data/sarcasm.json

import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def reading_file(filename):
    with open(filename, "r") as f:
        filedata = json.load(f)

    sentences = []
    labels = []

    for item in filedata:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    return sentences, labels

def tokenizing_data(train_sentences, test_sentences, corpus_size,\
                 max_length, trunc_type, padding_type, oov_tok):

    tokenizer = Tokenizer(num_words = corpus_size, oov_token = oov_tok)
    #use train_sentences to create the corpus
    tokenizer.fit_on_texts(train_sentences)
    # a dictionary of word:frequency
    word_index = tokenizer.word_index
    train_seq = tokenizer.texts_to_sequences(train_sentences)
    padded_train_seq = pad_sequences(train_seq, maxlen = max_length,
                    truncating = trunc_type, padding = padding_type)
    #create test_seq based on the corpus from train_sentences data
    test_seq = tokenizer.texts_to_sequences(test_sentences)
    padded_test_seq = pad_sequences(test_seq, maxlen = max_length,
                    truncating = trunc_type, padding = padding_type)

    reverse_word_index = dict([(value, key)
                            for (key, value) in word_index.items()])

    return padded_train_seq, padded_test_seq, reverse_word_index, tokenizer

def buiding_nn_model(padded_train_seq, padded_test_seq, 
                train_labels, test_labels, corpus_size, max_length,
                embedding_dim, num_epochs):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim,
            input_length = max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(padded_train_seq, train_labels,
                epochs = num_epochs,
                validation_data = (padded_test_seq, test_labels),
                verbose = 2)

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #evaluate the model
    model_evaluation(acc, val_acc, loss, val_loss)
    e = model.layers[0]
    weights = e.get_weights()[0]

    return weights, model

def writing_v_m_files(weights, reverse_word_index, corpus_size):
    vfile = open('data/sarcasm/vecs.tsv','w', encoding ='utf-8')
    mfile = open('data/sarcasm/meta.tsv','w', encoding ='utf-8')

    for num in range(1, corpus_size):
        word = reverse_word_index[num]
        embeddings = weights[num]
        mfile.write(word+'\n')
        vfile.write('\t'.join([str(x) for x in embeddings])+'\n')
    vfile.close()
    mfile.close()

def decode_review(text, reverse_word_index):
    #get(i,?): all the padded 'O's in the vectors will be converted to '?'
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

def model_evaluation(acc, val_acc, loss, val_loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc, 'r', "Training Accuracy")
    plt.plot(epochs,val_acc, 'b')
    plt.title("Training and validation accuracy")
    plt.show()

    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("Training and validation loss")
    plt.show()

def prediting_model(sentence, tokenizer, model, 
                            max_length, padding_type, trunc_type):
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, 
                        padding=padding_type, truncating=trunc_type)
    print(sentence)
    print(padded)
    classes = model.predict(padded)

    i =0
    while(i<len(sentence)):
        print(sentence[i])
        if classes[i]>0.5:
            print("It is sarcastic:", classes[i])
        else:
            print("It is not sarcastic:", classes[i])
        i += 1

########################################################################
# The main() function
def main():
    
    ### changable parameters:
    #specifies the maximum number of top most common words to be tokenized 
    corpus_size =10000
    # the number of dimensions for the vector representing the word encoding
    embedding_dim = 16 
    max_length = 100 #max length of the sentences
    trunc_type = 'post' #type of truncated: the tail of sentence
    padding_type='post' #type of padding: at tail of sequence
    oov_tok = "<OOV>" #value used for unidentified word in the corpus
    num_epochs = 10 #number of epochs for training
    train_size = 20000 

    filename = 'data/sarcasm.json'
    sentences, labels= reading_file(filename)

    #generate training and testing sets
    train_sentences = sentences[0:train_size]
    test_sentences = sentences[train_size:]
    train_labels = labels[0:train_size]
    test_labels = labels[train_size:]

    #tokenization
    padded_train_seq, padded_test_seq, reverse_word_index, tokenizer =\
                tokenizing_data(train_sentences, test_sentences,\
                corpus_size, max_length, trunc_type,padding_type, oov_tok)
    
    # print(train_sentences[1])
    # print(padded_train_seq[1])
    # print (decode_review(padded_train_seq[1], reverse_word_index))

    weights, model = buiding_nn_model(padded_train_seq, padded_test_seq, 
                train_labels, test_labels, corpus_size, max_length,
                embedding_dim, num_epochs)
    
    writing_v_m_files(weights, reverse_word_index, corpus_size)

    sentence = ["granny starting to fear spiders in the garden might be real", 
    "game of thrones season finale showing this sunday night"]
    
    prediting_model(sentence, tokenizer, model, 
                            max_length, padding_type, trunc_type)

    # Viewing the words and their vectors with Embedding Projector 
    # https://projector.tensorflow.org/
#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

