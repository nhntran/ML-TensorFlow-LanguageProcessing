
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

import os
import io
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def imdb_data_import():
    imdb, infor = tfds.load("imdb_reviews", with_info = True,
                                as_supervised = True)
    train_data, test_data = imdb['train'], imdb['test']

    ##convert data into numpy array for tokenizer
    train_sentences = []; train_labels = []
    test_sentences = []; test_labels = []
    for s, l in train_data:
        train_sentences.append(str(s.numpy()))
        train_labels.append(l.numpy())

    for s,l in test_data:
        test_sentences.append(str(s.numpy()))
        test_labels.append(l.numpy())

    return train_sentences, np.array(train_labels),\
                test_sentences, np.array(test_labels),

def tokenizing_data(train_sentences, test_sentences, corpus_size, max_length):
    # changable parameters:
    trunc_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words = corpus_size, oov_token = oov_tok)
    #use train_sentences to create the corpus
    tokenizer.fit_on_texts(train_sentences)
    # a dictionary of word:frequency
    word_index = tokenizer.word_index
    train_seq = tokenizer.texts_to_sequences(train_sentences)
    padded_train_seq = pad_sequences(train_seq, maxlen = max_length,
                    truncating = trunc_type)
    #create test_seq based on the corpus from train_sentences data
    test_seq = tokenizer.texts_to_sequences(test_sentences)
    padded_test_seq = pad_sequences(test_seq, maxlen = max_length)
    reverse_word_index = dict([(value, key)
                            for (key, value) in word_index.items()])

    return padded_train_seq, padded_test_seq, reverse_word_index

def buiding_nn_model(padded_train_seq, padded_test_seq, 
                train_labels, test_labels, corpus_size, max_length):
    # changable parameters:
    embedding_dim = 16
    num_epochs = 10

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim,
            input_length = max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics = ['accuracy'])
    model.summary()
    model.fit(padded_train_seq, train_labels,
                epochs = num_epochs,
                validation_data = (padded_test_seq, test_labels))
    e = model.layers[0]
    weights = e.get_weights()[0]

    return weights

def writing_v_m_files(weights, reverse_word_index, corpus_size):
    vfile = open('data/imdb/vecs.tsv','w', encoding ='utf-8')
    mfile = open('data/imdb/meta.tsv','w', encoding ='utf-8')

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

########################################################################
# The main() function
def main():
    
    # changable parameters:
    corpus_size =10000
    max_length = 120

    train_sentences, train_labels, test_sentences,\
                        test_labels = imdb_data_import()
    
    padded_train_seq, padded_test_seq, reverse_word_index =\
                tokenizing_data(train_sentences, test_sentences,\
                corpus_size, max_length)
    
    # print(train_sentences[1])
    # print(padded_train_seq[1])
    # print (decode_review(padded_train_seq[1], reverse_word_index))

    weights = buiding_nn_model(padded_train_seq, padded_test_seq, 
                train_labels, test_labels, corpus_size, max_length)
    
    writing_v_m_files(weights, reverse_word_index, corpus_size)

    # Viewing the words and their vectors with Embedding Projector 
    # https://projector.tensorflow.org/
#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

