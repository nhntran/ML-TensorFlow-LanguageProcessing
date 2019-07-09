Deep Learning with TensorFlow - Language Processing - Tutorials
================
Codes courtesy from TensorFlow in Practice Specialization by deeplearning.ai on Coursera, modified by Tran Nguyen

-   [1. Text Processing with Tokenizer](#text-processing-with-tokenizer)
-   [2.Text Processing for .json Data](#text-processing-for-.json-data)
-   [3. Tokenizing the BBC News Dataset](#tokenizing-the-bbc-news-dataset)

Quick notes from the courses + codes to run in Mac terminal. If you want to learn more about TensorFlow, check out the great courses in the "TensorFlow in Practice Specialization" by deeplearning.ai on Coursera.

The codes work well with TensorFlow 2.0

``` bash
pip install tensorflow==2.0.0-alpha0
```

Ref: <https://keras.io/preprocessing/text/>

#### 1. Text Processing with Tokenizer

-   Codes: tokenizer\_basics.py
-   What you will learn: (i) Generating a corpus from a list of sentences using Tokenizer. (ii) Converting a list of sentences into a sequence of word index from the corpus. (iii) Using padding to get the list of all the sequences that have the same size.

#### 2.Text Processing for .json Data

-   Codes: text\_processing\_json.py
-   What you will learn: (i) Getting data from json file and tokenize the dataset.

#### 3. Tokenizing the BBC News Dataset

-   Codes: text\_processing\_bcc\_data.py
-   Input: a folder 'bbc' contains 5 folders of 5 different classes (business, entertainment, politics, sport, tech). Each class has mutiple text files of news.
-   Ouput: a corpus for each class, and all the sequences of word-index generated from specific corpus for each of the 5 classes.
-   What you will learn: (i) Reading multiple files from folder. (ii) Tokenizing a dataset with removing a list of common stopwords.
