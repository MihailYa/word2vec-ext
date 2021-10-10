import pickle
import uuid
import os
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecgreadydataset import EcgReadyDataset
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ecgdetectors import Detectors
from gensim.models import Word2Vec
from datetime import datetime

_mit_abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V",
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]


class EcgDataset:

    @staticmethod
    def _classify_beat(symbol):
        if symbol in _mit_abnormal_beats:
            return 1
        elif symbol == "N" or symbol == ".":
            return 0

    @staticmethod
    def _get_sets_names(dir_path):
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

        def get_file_name(file):
            return file.split(".", 1)[0]

        data = [int(x) for x in list(set(list(map(get_file_name, onlyfiles))))]
        data.sort()
        return data

    @staticmethod
    def _combine_sets_beats_and_features(records_path, sets_numbers):
        sets_data = []
        for set_number in sets_numbers:
            print("Loading set: " + str(set_number))
            beats, labels, normal_percentage = EcgDataset._load(records_path, set_number)
            sets_data.append({
                "beats": beats,
                "labels": labels,
                "normal_percentage": normal_percentage
            })
        return sets_data

    @staticmethod
    def _get_sequence(signal, beat_loc, window_sec, fs):
        window_one_side = window_sec * fs
        beat_start = beat_loc - window_one_side
        beat_end = beat_loc + window_one_side
        if beat_end < signal.shape[0]:
            sequence = signal[beat_start:beat_end, 0]
            return sequence
        else:
            return np.array([])

    @staticmethod
    def _load(out_dir, record_number):
        record = wfdb.rdrecord(str(out_dir) + '/' + str(record_number))
        annotation = wfdb.rdann(str(out_dir) + '/' + str(record_number), "atr")
        atr_symbols = annotation.symbol
        atr_samples = annotation.sample
        fs = record.fs
        scaler = StandardScaler()
        signal = scaler.fit_transform(record.p_signal)

        # r_peaks = define_r_peaks_indices(record_wave)
        # heartbeat_len = define_heartbeat_len(r_peaks)
        # beats = split_in_beats(record_wave, heartbeat_len, r_peaks)
        labels = []
        valid_beats = []
        window_sec = 3
        for i, i_sample in enumerate(atr_samples):
            label = EcgDataset._classify_beat(atr_symbols[i])
            if label is not None:
                sequence = EcgDataset._get_sequence(signal, i_sample, window_sec, fs)
                if sequence.size > 0:
                    labels.append(label)
                    valid_beats.append(sequence)

        normal_percentage = sum(labels) / len(labels)

        assert len(valid_beats) == len(labels)
        return valid_beats, labels, normal_percentage

    @staticmethod
    def _download_dataset(out_dir, reload):
        if not os.path.isdir(out_dir) or reload:
            wfdb.dl_database('mitdb', out_dir)

    @staticmethod
    def _load_and_save_set(records_path, dataframe_path, reload=False):
        EcgDataset._download_dataset(records_path, reload)
        if os.path.exists(dataframe_path) and not reload:
            return pd.read_pickle(dataframe_path)
        else:
            print("Reloading dataset")
            data_frame = pd.DataFrame(
                EcgDataset._combine_sets_beats_and_features(records_path, EcgDataset._get_sets_names(records_path)))
            data_frame.to_pickle(dataframe_path)
            return data_frame

    @staticmethod
    def cache_from_mit(mit_records_path, dataframe_path, reload=False):
        data_frame = EcgDataset._load_and_save_set(mit_records_path, dataframe_path, reload)
        return EcgDataset(data_frame)

    @staticmethod
    def prepare_train_and_test(mit_records_path, dataframe_path,
                               train_path, test_path,
                               reload_base_data=False, test_size=0.25,
                               random_state=42, sets_count_limit=None, reload_train_test=False):
        if reload_train_test or not os.path.exists(train_path) or not os.path.exists(test_path):
            ecgdataset = EcgDataset.cache_from_mit(mit_records_path, dataframe_path, reload_base_data)

            train, test = ecgdataset.split_train_test(test_size=test_size, random_state=random_state)

            train_ready = train.concatenate_datasets()
            test_ready = test.concatenate_datasets()

            train_ready.save(train_path)
            test_ready.save(test_path)
        else:
            return EcgReadyDataset.load(train_path), EcgReadyDataset.load(test_path)

    dataframe: pd.DataFrame

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def split_train_test(self, test_size=0.25, random_state=42, sets_count_limit=None):
        data_frame = self.dataframe
        if sets_count_limit is not None:
            data_frame = data_frame[:sets_count_limit]
        print("Splitting dataset")
        bins = [0, 0.2, 0.6, 1.0]
        data_frame["bin"] = pd.cut(data_frame['normal_percentage'], bins=bins, labels=False, include_lowest=True)
        train, validation = train_test_split(data_frame, test_size=test_size, stratify=data_frame["bin"],
                                             random_state=random_state)

        return EcgDataset(train), EcgDataset(validation)

    def get_concatenated_rows(self):
        """

        :param dataframe:
        :return: dataframe with connected rows
        """
        all_beats = []
        all_labels = []
        for i, row in self.dataframe.iterrows():
            all_beats.append(row["beats"])
            all_labels.append(row["labels"])

        beats = np.concatenate(all_beats).tolist()
        labels = np.concatenate(all_labels).tolist()
        return pd.DataFrame({
            "beats": beats,
            "labels": labels
        })

    def concatenate_datasets(self):
        concated_df = self.get_concatenated_rows()
        return EcgReadyDataset(concated_df)
