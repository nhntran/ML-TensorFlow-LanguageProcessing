
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
#/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/
#MachineLearning/TensorFlow_LanguageProcessing/

# Download data
# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
#     -O data/sarcasm.json

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def reading_file(filename):
    with open(filename, "r") as f:
        filedata = json.load(f)

    sentences = []
    labels = []
    urls = []

    for item in filedata:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

    return sentences, labels, urls

def text_processing(sentences):

    #tokenizer = Tokenizer(oov_token="<OOV>", num_words = 100,
    #            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    # num_words: the maximum number of words to keep, based on word frequency. 
    # Only the most common num_words-1 words will be kept.
    # if want to keep -, for example "twenty-one": delete - in the filters

    tokenizer = Tokenizer(oov_token="<OOV>")
    # specify a token to use for unknown words as OOV
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding = 'post')
    #default of pad_sequences:
    #They’ll get padded to the length of the longest sequence 
    #by adding zeros to the beginning of shorter ones
    print(padded[0])
    print(padded.shape)


########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    filename = 'data/sarcasm.json'
    sentences, labels, urls = reading_file(filename)
    text_processing(sentences)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

