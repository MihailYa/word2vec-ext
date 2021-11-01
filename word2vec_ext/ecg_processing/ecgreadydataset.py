import pickle
import uuid
import os
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ecgdetectors import Detectors
from gensim.models import Word2Vec
from datetime import datetime


class EcgReadyDataset:
    @staticmethod
    def load(file_path):
        df = pd.read_pickle(file_path)
        train_start_indices = pd.read_pickle(file_path + ".start_indices.pkl")
        return EcgReadyDataset(df, train_start_indices)

    dataframe: pd.DataFrame
    train_start_indices: pd.DataFrame

    def __init__(self, dataframe, train_start_indices):
        self.dataframe = dataframe
        self.train_start_indices = train_start_indices

    def save(self, file_path):
        self.dataframe.to_pickle(file_path)
        self.dataframe.to_pickle(file_path + ".start_indices.pkl")

    def get_beats_and_labels(self):
        return self.dataframe["beats"].tolist(), self.dataframe["labels"].tolist()
