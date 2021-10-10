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
        return EcgReadyDataset(df)

    dataframe: pd.DataFrame

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def save(self, file_path):
        self.dataframe.to_pickle(file_path)

    def get_beats_and_labels(self):
        return self.dataframe["beats"].tolist(), self.dataframe["labels"].tolist()
