import pickle
import uuid
import os
import wfdb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ecgdetectors import Detectors
from gensim.models import Word2Vec
from datetime import datetime


class Word2VecExt:
    @staticmethod
    def load(file_path):
        word2vec = Word2Vec.load(file_path)
        return Word2VecExt(word2vec)

    @staticmethod
    def load_or_fit_and_save(file_path, sentences, workers=3, vector_size=100, min_count=5, window=5, sample=1e-3,
                             reset=False):
        if not (os.path.exists(file_path) and reset):
            model = Word2Vec(sentences, workers=workers,
                             vector_size=vector_size, min_count=min_count,
                             window=window, sample=sample)
            model.save(file_path)
        else:
            model = Word2Vec.load(file_path)
        return Word2VecExt(model)

    @staticmethod
    def load_or_fit_words_and_save(words, sentences_start_indices, file_path, workers=3, vector_size=100,
                                   min_count=5, window=5, sample=1e-3, sg=0, hs=0, negative=5,
                                   alpha=0.025,
                                   reset=False):
        if not os.path.exists(file_path) or reset:
            print("TeachWord2Vec: Concatenating sentences")
            sentences = []
            for i in range(len(sentences_start_indices) - 1):
                start_index = sentences_start_indices[i]
                end_index = sentences_start_indices[i + 1]
                sentences.append(words[start_index:end_index])
            sentences.append(words[sentences_start_indices[-1]:])

            print("TeachWord2Vec: Start Word2Vec training")
            model = Word2Vec(sentences, workers=workers,
                             vector_size=vector_size, min_count=min_count,
                             window=window, sample=sample, sg=sg, hs=hs, negative=negative, alpha=alpha)
            model.save(file_path)
        else:
            print("Using cached Word2Vec from file: ", file_path)
            model = Word2Vec.load(file_path)

        return Word2VecExt(model)

    word2vec: Word2Vec

    def __init__(self, word2vec=None):
        self.word2vec = word2vec

    def save(self, file_path):
        self.word2vec.save(file_path)

    def vectorize_valid(self, words):
        feature_vecs = np.zeros((len(words), self.word2vec.vector_size), dtype='float32')

        index2word_set = set(self.word2vec.wv.index_to_key)

        indices = []
        valid_words_count = 0
        for i in range(len(words)):
            cur_word = words[i]
            if cur_word in index2word_set:
                feature_vecs[valid_words_count] = self.word2vec.wv[cur_word]
                valid_words_count = valid_words_count + 1
                indices.append(i)

        return indices, feature_vecs[:valid_words_count]

    def vectorize_valid_with_labels(self, words, labels):
        valid_beats_indices, features = self.vectorize_valid(words)

        valid_beats_labels = [labels[i] for i in valid_beats_indices]
        print("Valid labels count:")
        print(np.bincount(np.array(valid_beats_labels)))

        return features, valid_beats_labels
