
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

# Download the data
# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt \
#     -O data/sonnets.txt


import tensorflow as tf
import tensorflow_datasets as tfds                        
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

def data_import(filename):
    
    tokenizer = Tokenizer()
    data = open(filename).read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    corpus_size = len(tokenizer.word_index)+1
    print(corpus_size)
    test = tokenizer.texts_to_sequences([corpus[0]])
    
    sequences =[]
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            n_gram_sequence = token_list[:i+1]
            sequences.append(n_gram_sequence)
    max_len = max([len(x) for x in sequences])
    #print(len(sequences))
    sequences = np.array(pad_sequences(sequences,maxlen=max_len, padding = 'pre'))
    train_sequences, labels = sequences[:,:-1], sequences[:,-1]
    train_labels = tf.keras.utils.to_categorical(labels, num_classes = corpus_size)

    return tokenizer, train_sequences, train_labels, corpus_size, max_len

def buiding_nn_model(corpus_size, embedding_dim, max_len,
                            train_sequences, train_labels, num_epochs):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim,
                                    input_length = max_len-1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
        tf.keras.layers.Dense(corpus_size, activation = 'softmax'),
        ])
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.01),
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_sequences, train_labels, epochs = num_epochs,
                                verbose = 1)
    #after training, saving the model into the .h5 file
    model.save('TF_NLP_text_prediction_Shakespeare.h5') 

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    loss = history.history['loss']

    #evaluate the model
    model_evaluation(acc, loss)

    return model

def buiding_nn_model_optimization(corpus_size, embedding_dim, max_len,
                    train_sequences, train_labels, num_epochs, drop_out_val):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(corpus_size, embedding_dim,
                                    input_length = max_len-1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, 
                                        return_sequences = True)),
        tf.keras.layers.Dropout(drop_out_val),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
        tf.keras.layers.Dense((corpus_size/2), activation = 'relu',
            kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(corpus_size, activation = 'softmax'),
        ])
    model.compile(loss = 'categorical_crossentropy', optimizer = "adam",
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_sequences, train_labels, epochs = num_epochs,
                                verbose = 1)
    #after training, saving the model into the .h5 file
    model.save('TF_NLP_text_prediction_Shakespeare_optimization.h5') 

    ## retrieve accuracy and loss values
    acc = history.history['accuracy']
    loss = history.history['loss']

    #evaluate the model
    model_evaluation(acc, loss)

    return model

def model_evaluation(acc, loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc)
    plt.title("Training accuracy")
    plt.xlabel ("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(epochs,loss)
    plt.title("Training loss")
    plt.xlabel ("Epochs")
    plt.ylabel("Loss")
    plt.show()

def text_prediction(seed_text, next_words, max_len, model, tokenizer):
    print("Seedtext:", seed_text)
    # print("Length:",next_words)
    # print("MaxLength:",max_len)
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        token_list = pad_sequences([token_list],
                        maxlen = max_len-1, padding='pre')
        # print(token_list)
        predicted = model.predict_classes(token_list, verbose =0)
        output_word=""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word=word
                break
        seed_text +=" "+output_word
    print("Predicted text:", seed_text)

########################################################################
# The main() function
def main():
    
    ### changable parameters:
    # dimensions for the vector representing the subwords encoding
    embedding_dim = 100
    num_epochs = 100 #number of epochs for training
    drop_out_val = 0.2 #drop out 20%
    filename = 'data/sonnets.txt'
    seed_text = "Help me Obi Wan Kenobi, you're my only hope"

    tokenizer, train_sequences, train_labels,\
                        corpus_size, max_len = data_import(filename)
    
    # model = buiding_nn_model(corpus_size, embedding_dim,
    #                max_len, train_sequences, train_labels, num_epochs)
    
    #model optimization
    model_optimization = buiding_nn_model_optimization(corpus_size, embedding_dim,
                max_len, train_sequences, train_labels, num_epochs, drop_out_val)
    #Second time running: Loading the model again
    #new_model = tf.keras.models.load_model('TF_NLP_text_prediction_Shakespeare.h5')

    #prediction
    next_words =100
    #length of the prediction word: set equal to max_len
    # seed_list = seed_text.split()
    # if (max_len - len(seed_list)>0):
    #     next_words = max_len - len(seed_list)
    # else: 
    #     next_words = 10

    text_prediction(seed_text, next_words, max_len, model, tokenizer)



#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

