import pandas as pd


class EcgDataset:
    @staticmethod
    def load(file_path):
        df = pd.read_pickle(file_path)
        train_start_indices = pd.read_pickle(file_path + ".start_indices.pkl")
        return EcgDataset(df, train_start_indices)

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
