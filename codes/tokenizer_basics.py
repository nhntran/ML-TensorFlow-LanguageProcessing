
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
#/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/
#MachineLearning/TensorFlow_LanguageProcessing/


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

### ***** GENERATING CORPUS *****

sentences =[
'Mom loves me',
'I love mom',
'Dad loves me']

tokenizer = Tokenizer() # create an Tokenizer object
#default filter: remove all the non-alphabet letters.
#filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#lowercase for all the words
tokenizer.fit_on_texts(sentences) 
#generate a corpus of word: occurring frequency from a list of sentences

word = tokenizer.word_index #extract the dictionary
print(word)
# => {'mom': 1, 'loves': 2, 'me': 3, 'i': 4, 'love': 5, 'dad': 6}

### ***** CREATING WORD INDEX FROM SENTENCES ***** 

sequences = tokenizer.texts_to_sequences(sentences) 
#change sentences into index value
print(sequences) 
# => [[1, 2, 3], [4, 5, 1], [6, 2, 3]]

sentences2=[
'I love my dad',
'My sister loves me',
'I love my mom, my dad and my sister']
sequences2 = tokenizer.texts_to_sequences(sentences2) 
print(sequences2)
#=> [[4, 5, 6], [2, 3], [4, 5, 1, 6]]: skip all unidentified words!!!

tokenizer2 = Tokenizer(oov_token = "<OOV>")
##oov: out of vocabulary
tokenizer2.fit_on_texts(sentences)
word = tokenizer2.word_index #extract the dictionary
print(word)
#{'<OOV>': 1, 'mom': 2, 'loves': 3, 'me': 4, 'i': 5, 'love': 6, 'dad': 7}
sequences2 = tokenizer2.texts_to_sequences(sentences2) 
print(sequences2)
#[[5, 6, 1, 7], [1, 1, 3, 4], [5, 6, 1, 2, 1, 7, 1, 1, 1]]
#unidentified words as index 1.

### ***** PADDING: ALL THE SENTENCES HAVE THE SAME SIZE OF WORD INDEX ***** 

padding = pad_sequences(sequences2)
# default: put O before sentences
print(padding)
#[[0 0 0 0 0 5 6 1 7]
# [0 0 0 0 0 1 1 3 4]
# [5 6 1 2 1 7 1 1 1]]

padding1 = pad_sequences(sequences2, padding = 'post',
            truncating = 'post', maxlen = 5)
#padding = 'post': put O after sentences
#maxlen =5: truncated to have all sentences of size 5
# => default: lose values from the beginning of the sentences
# => can change to truncating = 'post' => lose values from the end
print(padding1)
#[[5 6 1 7 0]
# [1 1 3 4 0]
# [5 6 1 2 1]]
print(padding1.shape)
#(3,5)

print(sentences2[2])
print(padding1[2])
#I love my mom, my dad and my sister
#[5 6 1 2 1] #truncated version
