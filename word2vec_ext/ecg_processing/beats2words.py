import pickle
import os
import numpy as np
from sklearn.cluster import KMeans


class Beats2Words:
    qrs_kmeans = None
    pt_kmeans = None

    def save(self, file_name):
        pickle.dump(self.qrs_kmeans, open(file_name + "_qrs.pkl", "wb"))
        pickle.dump(self.pt_kmeans, open(file_name + "_pt.pkl", "wb"))

    def load(self, file_name):
        self.qrs_kmeans = pickle.load(open(file_name + "_qrs.pkl", "rb"))
        self.pt_kmeans = pickle.load(open(file_name + "_pt.pkl", "rb"))

    @staticmethod
    def _split_beats_in_p_qrt_t(beats_list):
        heart_beat_len = len(beats_list[0])
        retreat_value = heart_beat_len // 2
        c_p_waves = []
        c_qrs_waves = []
        c_t_waves = []
        for j in range(len(beats_list)):
            c_p_waves.append(beats_list[j][:retreat_value - 15])
            c_qrs_waves.append(beats_list[j][retreat_value - 15:retreat_value + 15])
            c_t_waves.append(beats_list[j][retreat_value + 15:])
        return c_p_waves, c_qrs_waves, c_t_waves

    # Convert PT cluster predictions to letters
    @staticmethod
    def _convert_pt_predictions_to_letters(predictions):
        alphabet_for_pt = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i',
                           9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
                           18: 's', 19: 't'}

        def get_symbol_pt(x):
            return alphabet_for_pt[x]

        vfunc = np.vectorize(get_symbol_pt)

        return vfunc(predictions)

    # Convert QRS clusters predictions to letters
    @staticmethod
    def _convert_qrs_predictions_to_letters(predictions):
        alphabet_for_qrs = {0: 'u', 1: 'v', 2: 'w', 3: 'x', 4: 'y', 5: 'z'}

        def get_symbol_qrs(x):
            return alphabet_for_qrs[x]

        vfunc_2 = np.vectorize(get_symbol_qrs)

        return vfunc_2(predictions)

    @staticmethod
    def _join_waves_letters_to_words(qrs_letters, pt_letters):
        words = []
        signal_half_len = len(qrs_letters)//2
        for i in range(len(qrs_letters)):
            word = ''
            word += pt_letters[i]
            word += qrs_letters[i]
            # T signal is encoded in second half
            word += pt_letters[i + signal_half_len]
            words.append(word)

        return words

    def fit_and_predict_words(self, beats_list, cache_file_name, reset_cache=False):

        if reset_cache or not (os.path.exists(cache_file_name + "_qrs.pkl")
                               and os.path.exists(cache_file_name + "_pt.pkl")):
            (p_waves, qrs, t_waves) = Beats2Words._split_beats_in_p_qrt_t(beats_list)
            print("Fitting QRS KMeans")
            self.qrs_kmeans = KMeans(init='k-means++', n_clusters=6, n_init=3)
            # self.qrs_kmeans = KMeans(init='k-means++', n_clusters=50, n_init=25, max_iter=600)
            self.qrs_kmeans.fit(qrs)

            qrs_predicted = self.qrs_kmeans.predict(qrs)

            print("Fitting PT KMeans")
            c_pt_waves = np.array(p_waves + t_waves)
            self.pt_kmeans = KMeans(init='k-means++', n_clusters=20, n_init=10)
            # self.pt_kmeans = KMeans(init='k-means++', n_clusters=200, n_init=100, max_iter=600)
            self.pt_kmeans.fit(c_pt_waves)

            print("Saving BeatsToWordsConverter to cache: ", cache_file_name)
            self.save(cache_file_name)

            pt_predicted = self.pt_kmeans.predict(c_pt_waves)

            qrs_letters = Beats2Words._convert_qrs_predictions_to_letters(qrs_predicted)
            pt_letters = Beats2Words._convert_pt_predictions_to_letters(pt_predicted)
            return Beats2Words._join_waves_letters_to_words(qrs_letters, pt_letters)
        else:
            print("Using cached BeatsToWordsConverter from file: ", cache_file_name)
            self.load(cache_file_name)
            return self.predict_words(beats_list)

    def predict_words(self, beats_list):
        (p_waves, qrs, t_waves) = Beats2Words._split_beats_in_p_qrt_t(beats_list)
        c_pt_waves = np.array(p_waves + t_waves)
        qrs_predicted = self.qrs_kmeans.predict(qrs)
        pt_predicted = self.pt_kmeans.predict(c_pt_waves)

        qrs_letters = Beats2Words._convert_qrs_predictions_to_letters(qrs_predicted)
        pt_letters = Beats2Words._convert_pt_predictions_to_letters(pt_predicted)
        return Beats2Words._join_waves_letters_to_words(qrs_letters, pt_letters)
