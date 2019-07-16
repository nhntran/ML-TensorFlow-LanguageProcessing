
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
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
#    -O /tmp/irish-lyrics-eof.txt


import tensorflow as tf
import tensorflow_datasets as tfds                        
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def data_import(filename):
    
    tokenizer = Tokenizer()
    data = open(filename).read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    corpus_size = len(tokenizer.word_index)+1
    #print(corpus_size)
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
        # output shape with 64 unit: (None, 128) 
        tf.keras.layers.Dense(corpus_size, activation = 'softmax'),
        ])
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.01),
                    metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_sequences, train_labels, epochs = num_epochs,
                                verbose = 1)
    #after training, saving the model into the .h5 file
    #model.save('TF_NLP_text_prediction.h5') 

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
    filename = 'data/irish-lyrics.txt'
    seed_text = "I've got a good feeling"

    tokenizer, train_sequences, train_labels,\
                        corpus_size, max_len = data_import(filename)
    
    model = buiding_nn_model(corpus_size, embedding_dim,
                   max_len, train_sequences, train_labels, num_epochs)
    # Second time running: Loading the model again
    # new_model = tf.keras.models.load_model('TF_NLP_text_prediction.h5')

    #prediction
    #length of the prediction word: set equal to max_len
    seed_list = seed_text.split()
    next_words = max_len - len(seed_list)

    text_prediction(seed_text, next_words, max_len, new_model, tokenizer)



#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

