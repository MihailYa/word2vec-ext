from word2vec_ext.ecg_processing import ecgdatasetsholder
from word2vec_ext.ecg_processing import beats2words
from word2vec_ext.word2vec import word2vecext

import uuid
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

if __name__ == '__main__':
    rsc_dir = "../example_rsc"
    test_start_time = datetime.now()

    ecgdataset = ecgdatasetsholder.EcgDatasetsHolder.cache_from_mit(
        sets_count_limit=7,
        database_name="mitdb",
        mit_records_path=rsc_dir + "/mit_records",
        dataframe_path=rsc_dir + "/ecgdataset_without_abnormal",
        annotator_type="symbol",
        reload=False
    )

    train, test = ecgdataset.split_train_test(test_size=0.25, random_state=42)

    train_ready = train.concatenate_datasets()
    test_ready = test.concatenate_datasets()

    train_ready.save(rsc_dir + "/train_dataset7")
    test_ready.save(rsc_dir + "/test_dataset7")

    # All up code is equal to code below:
    # train_ready, test_ready = ecgdataset.prepare_train_and_test(rsc_dir + "/mit_records", rsc_dir + "/ecgdataset",
    #                                                             rsc_dir + "/train_dataset",
    #                                                             rsc_dir + "/test_dataset", test_size=0.25, random_state=42)

    data_preparing_start_time = datetime.now()

    beats2wordsModel = beats2words.Beats2Words()
    train_words = beats2wordsModel.fit_and_predict_words(
        beats=train_ready.dataframe["beats"].tolist(),
        cache_file_name=rsc_dir + "/beats2words7",
        reset_cache=False)

    validation_words = beats2wordsModel.predict_words(test_ready.dataframe["beats"].tolist())

    num_features = 300
    word2vecExtModel = \
        word2vecext.Word2VecExt.load_or_fit_words_and_save(
            words=train_words,
            sentences_start_indices=train_ready.record_start_indices["start_indices"].tolist(),
            file_path=rsc_dir + "/word2vec7",
            vector_size=num_features,
            reset=False)
    train_data = word2vecExtModel.vectorize_valid_with_labels(
        words=train_words,
        labels=train_ready.dataframe["labels"].tolist())

    validation_data = word2vecExtModel.vectorize_valid_with_labels(
        validation_words,
        test_ready.dataframe["labels"].tolist())

    (train_x, train_y), (validation_x, validation_y) = (train_data, validation_data)

    data_preparing_time_spend = datetime.now() - data_preparing_start_time

    print("Training random forest")
    training_start_time = datetime.now()
    forest = RandomForestClassifier()
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
    f.write("Training with word2vec (without cache)")
    f.write(report)
    f.write("\n===========================================")
    f.write("\n\n")
    f.close()
