## Description

This is a library with extensions for sklearn, gensim and wfdb libraries.  
This library provides more specific methods for ECG analysis.

## Examples

All the examples located in the `example` directory.

There are examples:

* `example/example.py` - example of library usage for:
  * downloading ECG dataset
  * splitting in train and test datasets
  * _Beats2Words_ for converting ECG beats into words using _kmeans_
  * _Word2VecExt_ for vectorizing words
  * training _RandomForestClassifier_ and outputting its predictions results
  
  To run execute: `python -m example.example`
* `example/example_comparing_speed.py` - example of library usage for comparing speed of training models
  using _Word2Vec_ and without _Word2Vec_

  To run execute: `python -m example.example_comparing_speed`
* `example/example2_predicting_surrounding_words.py` - example of library usage for predicting surrounding ECG-words
  using _Word2Vec_

  To run execute: `python -m example.example2_predicting_surrounding_words`
* `example/example3_textrank.py` - example of library usage for TextRank method applying on ECG-words

  To run execute: `python -m example.example3_textrank`
* `example/example4_predicting_synonym.py` - example of library usage for predicting synonym ECG-words using `Word2Vec`

  To run execute: `python -m example.example4_predicting_synonym`

Some of previous important runs located in `example/results.txt`

## Documentation
Documentation can be found in [ІП02-мп_Язенок.pdf](docs/ІП02-мп_Язенок.pdf)

## Jupyter notebook drafts
You can also use jupyter notebook drafts, which were used as a base for this library.  
Drafts located in the separate repository:
[https://github.com/MihailYa/word2vec-analysis](https://github.com/MihailYa/word2vec-analysis)
