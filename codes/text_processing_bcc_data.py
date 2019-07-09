
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
#/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/
#MachineLearning/TensorFlow_LanguageProcessing/

# Data:
# 2225 documents from the BBC news website, 
#stories in five topical areas from 2004-2005:
# business, entertainment, politics, sport, tech
# Ref: D. Greene and P. Cunningham. "Practical Solutions to the Problem of 
# Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
#Download site:
# http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

#stop_word list: 
#https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def reading_file(local_path1, text1, text2, text3, text4, text5, stop_word_list):
    business = []
    business = read_multiple_file(local_path1+text1)
    new_business = stop_words_removed(business, stop_word_list)

    entertainment = []
    entertainment = read_multiple_file(local_path1+text2)
    new_entertainment = stop_words_removed(entertainment, stop_word_list)

    politics = []
    politics = read_multiple_file(local_path1+text3)
    new_politics = stop_words_removed(politics, stop_word_list)

    sport = []
    sport = read_multiple_file(local_path1+text4)
    new_sport = stop_words_removed(sport, stop_word_list)

    tech = []
    tech = read_multiple_file(local_path1+text5)
    new_tech= stop_words_removed(tech, stop_word_list)

    return new_business, new_entertainment, new_politics, new_sport, new_tech

def read_multiple_file(foldername):
    output_list=[]
    for file in os.listdir(foldername):
        fn = foldername+'/'+file
        try:
            with open(fn,'rb') as f:
                #UnicodeDecodeError: 'utf-8' codec can't decode byte 
                    #0xa3 in position 257: invalid start byte
                # => fixed using 'rb'
                for line in f:
                    if not line.isspace():
                        text = line.strip().decode("utf-8", "ignore")
                        output_list.append(text)
        except FileNotFoundError:
            print("Cannot find file", file)
    return output_list  


def text_processing(sentences):

    #tokenizer = Tokenizer(oov_token="<OOV>", num_words = 100,
    #            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    # num_words: the maximum number of words to keep, based on word frequency. 
    # Only the most common num_words-1 words will be kept.
    # if want to keep -, for example "twenty-one": delete - in the filters

    tokenizer = Tokenizer(oov_token="<OOV>", filters =
        '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    # specify a token to use for unknown words as OOV
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding = 'post')
    #default of pad_sequences:
    #They’ll get padded to the length of the longest sequence 
    #by adding zeros to the beginning of shorter ones
    return padded

def stop_words_removed (sentences, stop_word_list):
    new_sentences = []
    for text in sentences:
        for word in stop_word_list:
            word_found = " " + word + " "
            text = text.replace(word_found, " ")
            text = text.replace("  "," ")
        new_sentences.append(text)
    # print(sentences[2])
    # print(new_sentences[2])
    return new_sentences


########################################################################
# The main() function
def main():
    
    stop_word_list = [ "a", "about", "above", "after", "again", "against", "all", 
    "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", 
    "being", "below", "between", "both", "but", "by", "could", "did", 
    "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "he'd", 
    "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", 
    "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", 
    "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", 
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
    "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", 
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", 
    "what's", "when", "when's", "where", "where's", "which", "while", "who", 
    "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", 
    "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    local_path1 = 'data/bbc-data/bbc/'
    business, entertainment, politics, sport, tech =\
        reading_file(local_path1, 'business', 
            'entertainment', 'politics', 'sport', 'tech', stop_word_list)
    #corpus and sequences from the 'business" class
    
    padded_business = text_processing(business)
    print(business[0])
    print(padded_business.shape)

    #corpus and sequences of other classes:
    padded_entertainment = text_processing(entertainment)
    padded_politics = text_processing(politics)
    padded_sport = text_processing(sport)
    padded_tech = text_processing(tech)
    print(padded_entertainment.shape)
    print(padded_politics.shape)
    print(padded_sport.shape)
    print(padded_tech.shape)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

