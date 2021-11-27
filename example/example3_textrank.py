from word2vec_ext.ecg_processing import ecgdataset
from word2vec_ext.ecg_processing import ecgdatasetsholder
from word2vec_ext.ecg_processing import beats2words
from word2vec_ext.word2vec import word2vecext
import spacy
import pytextrank
from collections import Counter
import uuid
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime


def text_rank_test(c_validation_words, validation_sentences_indices, labels):
    print(validation_sentences_indices)
    print(len(c_validation_words))
    print()
    print()

    for i in range(len(validation_sentences_indices)):
        if i == len(validation_sentences_indices) - 1:
            end_index = len(c_validation_words)
        else:
            end_index = validation_sentences_indices[i + 1]
        start_index = validation_sentences_indices[i]

        current_sentence_words = c_validation_words[start_index:end_index]
        current_sentence_labels = labels[start_index:end_index]
        current_sentence_labels_count = dict(Counter(current_sentence_labels))
        print("Text labels count: ", current_sentence_labels_count)

        current_sentence = '\n'.join(current_sentence_words)
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        doc = nlp(current_sentence)
        print("Text summary: ")
        phrase = doc._.phrases[0]
        word = phrase.text
        labels_of_word = [current_sentence_labels[i] for i, x in enumerate(current_sentence_words) if x == word]
        estimated_labels_count = dict(Counter(labels_of_word))
        print("Word:", word, "; Related labels:", estimated_labels_count)
        print(phrase.rank, phrase.count)
        print()
        print()

        # Code below make no sense
        # print(current_sentence_labels_count)
        # real_relation = current_sentence_labels_count.get(0, 0) / (current_sentence_labels_count.get(1, 0) + current_sentence_labels_count.get(0, 0))
        # estimated_relation = estimated_labels_count.get(0, 0) / (estimated_labels_count.get(1, 0) + estimated_labels_count.get(0, 0))
        # print("Real relation:", real_relation, "Estimated relation:", estimated_relation)
        # print("Error:", (estimated_relation - real_relation) / real_relation)
        # print()
        # print()


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
    train_words = beats2wordsModel.fit_and_predict_words(
        beats=train_ready.dataframe["beats"].tolist(),
        cache_file_name=rsc_dir + "/beats2words_test_for_5_datasets",
        reset_cache=True)

    text_rank_test(train_words, train_ready.record_start_indices["start_indices"].tolist(),
                   train_ready.dataframe["labels"].tolist())


if __name__ == '__main__':
    main()
