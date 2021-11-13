from word2vec_ext.ecg_processing import ecgdataset
from word2vec_ext.ecg_processing import ecgdatasetsholder
from word2vec_ext.ecg_processing import beats2words
from word2vec_ext.word2vec import word2vecext

import uuid
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime


def calculate_word2vec_accuracy(c_word2vec: word2vecext.Word2Vec, c_validation_words, validation_sentences_indices):
    print(validation_sentences_indices)
    print(len(c_validation_words))
    true_cases = 0
    false_cases = 0
    index2word_set = set(c_word2vec.wv.index_to_key)

    for i in range(len(validation_sentences_indices)):
        if i == len(validation_sentences_indices) - 1:
            end_index = len(c_validation_words)
        else:
            end_index = validation_sentences_indices[i + 1]
        start_index = validation_sentences_indices[i]
        for j in range(start_index, end_index - 1):  # end_index - 1 to compare with next word
            # Iterate over each word in sentences
            if not (c_validation_words[j] in index2word_set and c_validation_words[j + 1] in index2word_set):
                # Skip words pairs which are not in the dictionary
                continue
            word_to_vec_list = c_word2vec.wv.most_similar(positive=[c_validation_words[j]])
            similar_words = map(lambda it: it[0], word_to_vec_list)

            if c_validation_words[j + 1] in similar_words:
                true_cases = true_cases + 1
            else:
                false_cases = false_cases + 1

    print("True_cases =", true_cases)
    print("False_cases =", false_cases)
    print("True_cases/total_cases:", true_cases / (true_cases + false_cases))


def main():
    rsc_dir = "../example_rsc"

    ecgdataset = ecgdatasetsholder.EcgDatasetsHolder.cache_from_mit(
        sets_count_limit=5,
        database_name="mitdb",
        mit_records_path=rsc_dir + "/mit_records",
        dataframe_path=rsc_dir + "/ecgdataset",
        annotator_type="symbol",
        reload=False
    )

    train_ready = ecgdataset.concatenate_datasets()

    train_ready.save(rsc_dir + "/full_dataset")

    beats2wordsModel = beats2words.Beats2Words()
    train_words = beats2wordsModel.fit_and_predict_words(train_ready.dataframe["beats"].tolist(),
                                                         rsc_dir + "/beats2words_test_for_5_datasets",
                                                         reset_cache=True)

    num_features = 300
    word2vecExtModel = \
        word2vecext.Word2VecExt.load_or_fit_words_and_save(train_words,
                                                           train_ready.train_start_indices["start_indices"].tolist(),
                                                           rsc_dir + "/word2vec_5_dataset_test",
                                                           vector_size=num_features,
                                                           sg=1,
                                                           sample=1e-3,
                                                           window=10,
                                                           alpha=0.01,
                                                           reset=True)

    calculate_word2vec_accuracy(word2vecExtModel.word2vec, train_words, train_ready.train_start_indices["start_indices"].tolist())
    return
    # noinspection DuplicatedCode
    train_data = word2vecExtModel.vectorize_valid_with_labels(train_words, train_ready.dataframe["labels"].tolist())

    validation_data = word2vecExtModel.vectorize_valid_with_labels(validation_words, test_ready.dataframe["labels"].tolist())

    (train_x, train_y), (validation_x, validation_y) = (train_data, validation_data)

    data_preparing_time_spend = datetime.now() - data_preparing_start_time

    print("Training random forest")
    training_start_time = datetime.now()
    forest = GradientBoostingClassifier(n_estimators=500, learning_rate=0.001, max_depth=7)
    forest = forest.fit(train_x, train_y)
    training_time_spend = datetime.now() - training_start_time

    print("Predicting using random forest")
    result = forest.predict(validation_x)

    print("Creating classification report")
    repord_id = uuid.uuid1()
    report = str(classification_report(validation_y, result))
    print("Report id: " + str(repord_id))
    print(report)
    f = open("results.txt", "a")

    now = datetime.now()
    f.write("===========================================\n")
    f.write("Report id: " + str(repord_id) + "\n")
    f.write("Report start date: " + test_start_time.strftime("%d.%m.%Y %H:%M:%S") + "\n")
    f.write("Report end date: " + now.strftime("%d.%m.%Y %H:%M:%S") + "\n")
    f.write("Actual train size: " + str(len(train_x)) + "\n")
    f.write("Actual validation size: " + str(len(validation_x)) + "\n")
    f.write("Actual validation size: " + str(len(validation_x)) + "\n")
    f.write("Data preparing time millis: " + str(data_preparing_time_spend.total_seconds() * 1000) + "\n")
    f.write("Training working time millis: " + str(training_time_spend.total_seconds() * 1000) + "\n")
    f.write(report)
    f.write("\n===========================================")
    f.write("\n\n")
    f.close()


if __name__ == '__main__':
    main()
