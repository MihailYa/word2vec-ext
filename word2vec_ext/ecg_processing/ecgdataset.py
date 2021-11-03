import os
import wfdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from word2vec_ext.ecg_processing.ecgreadydataset import EcgReadyDataset

_mit_abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V",
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]

_mit_abnormal_aux = [
    "AFIB", "AFL", "J"
]


class EcgDataset:

    @staticmethod
    def _classify_beat(symbol):
        if symbol in _mit_abnormal_beats:
            return 1
        elif symbol == "N" or symbol == ".":
            return 0

    @staticmethod
    def _classify_aux(aux):
        if aux in _mit_abnormal_aux:
            return 1
        elif aux == "N" or aux == ".":
            return 0

    @staticmethod
    def _multi_classify_aux(aux):
        try:
            return _mit_abnormal_aux.index(aux) + 1
        except ValueError:
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
    def _combine_sets_beats_and_features(annotator_type, records_path, sets_numbers):
        """

        :param annotator_type: symbol/aux/aux_multi
        :param records_path:
        :param sets_numbers:
        :return:
        """
        sets_data = []
        for set_number in sets_numbers:
            print("Loading set: " + str(set_number))
            if os.path.exists(records_path + "/" + str(set_number) + ".dat"):
                beats, labels, normal_percentage = EcgDataset._load(annotator_type, records_path, set_number)
                if beats is not None:
                    sets_data.append({
                        "beats": beats,
                        "labels": labels,
                        "normal_percentage": normal_percentage
                    })
                else:
                    print("Skipping empty set #" + str(set_number))
            else:
                print("Set " + str(set_number) + " contains invalid data. Skipping")
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
    def _load(annotator_type, out_dir, record_number):
        record = wfdb.rdrecord(str(out_dir) + '/' + str(record_number))
        annotation = wfdb.rdann(str(out_dir) + '/' + str(record_number), "atr")
        atr_symbols = annotation.symbol
        atr_samples = annotation.sample
        atr_aux = [aux[1:] for aux in annotation.aux_note]
        fs = record.fs
        signal = record.p_signal

        labels = []
        valid_beats = []
        window_sec = 3
        aux_annotator_type = annotator_type == "aux"
        aux_multi_annotator_type = annotator_type == "aux_multi"
        for i, i_sample in enumerate(atr_samples):
            if aux_annotator_type:
                label = EcgDataset._classify_aux(atr_aux[i])
            elif aux_multi_annotator_type:
                label = EcgDataset._multi_classify_aux(atr_aux[i])
            else:
                label = EcgDataset._classify_beat(atr_symbols[i])
            if label is not None:
                sequence = EcgDataset._get_sequence(signal, i_sample, window_sec, fs)
                if sequence.size > 0:
                    labels.append(label)
                    valid_beats.append(sequence)
        if len(labels) == 0:
            return None, None, None

        if aux_multi_annotator_type:
            def map_abnormal(beat_label):
                if beat_label > 0:
                    return 0
                else:
                    return 1

            normal_percentage = sum(list(map(map_abnormal, labels))) / len(labels)
        else:
            normal_percentage = sum(labels) / len(labels)

        assert len(valid_beats) == len(labels)
        return valid_beats, labels, normal_percentage

    @staticmethod
    def _download_dataset(database_name, out_dir, reload):
        """

        :param database_name: afdb or mitdb
        :param out_dir:
        :param reload:
        :return:
        """
        if not os.path.isdir(out_dir) or reload:
            print("Downloading dataset")
            wfdb.dl_database(database_name, out_dir)

    @staticmethod
    def _load_and_save_set(database_name, records_path, dataframe_path, annotator_type, reload=False):
        """

        :param database_name: afdb or mitdb
        :param records_path:
        :param dataframe_path:
        :param annotator_type: symbol/aux/aux_multi
        :param reload:
        :return:
        """
        EcgDataset._download_dataset(database_name, records_path, reload)
        if os.path.exists(dataframe_path) and not reload:
            return pd.read_pickle(dataframe_path)
        else:
            print("Reloading dataset")
            data_frame = pd.DataFrame(
                EcgDataset._combine_sets_beats_and_features(annotator_type, records_path,
                                                            EcgDataset._get_sets_names(records_path)))
            data_frame.to_pickle(dataframe_path)
            return data_frame

    @staticmethod
    def cache_from_mit(sets_count_limit, database_name, mit_records_path, dataframe_path, annotator_type="symbol",
                       reload=False):
        """

        :param sets_count_limit:
        :param database_name: afdb or mitdb
        :param mit_records_path:
        :param dataframe_path:
        :param annotator_type: symbol/aux/aux_multi
        :param reload:
        :return:
        """
        data_frame = EcgDataset._load_and_save_set(database_name, mit_records_path, dataframe_path, annotator_type,
                                                   reload)
        if sets_count_limit is not None:
            data_frame = data_frame[:sets_count_limit]
        return EcgDataset(data_frame)

    @staticmethod
    def prepare_train_and_test(mit_records_path, dataframe_path,
                               train_path, test_path,
                               reload_base_data=False, test_size=0.25,
                               random_state=42, sets_count_limit=None, reload_train_test=False):
        if reload_train_test or not os.path.exists(train_path) or not os.path.exists(test_path):
            ecgdataset = EcgDataset.cache_from_mit(sets_count_limit, mit_records_path, dataframe_path, reload_base_data)

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
          :return: dataframe with connected rows
          """
        all_beats = []
        all_labels = []
        start_index = 0
        start_indices = []
        for i, row in self.dataframe.iterrows():
            start_indices.append(start_index)
            beats = row["beats"]
            all_beats.append(beats)
            all_labels.append(row["labels"])
            start_index = start_index + len(beats)

        beats = np.concatenate(all_beats).tolist()
        labels = np.concatenate(all_labels).tolist()
        return pd.DataFrame({
            "beats": beats,
            "labels": labels
        }), pd.DataFrame({"start_indices": start_indices})

    def concatenate_datasets(self):
        concated_df, train_start_indices = self.get_concatenated_rows()
        return EcgReadyDataset(concated_df, train_start_indices)
