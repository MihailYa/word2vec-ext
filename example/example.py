from word2vec_ext.word2vec import word2vecext
from word2vec_ext.ecg_processing import ecgdataset

rsc_dir = "../example_rsc"

ecgdataset = ecgdataset.EcgDataset.cache_from_mit(rsc_dir + "/mit_records", rsc_dir + "/ecgdataset")

train, test = ecgdataset.split_train_test(test_size=0.25, random_state=42)

train_ready = train.concatenate_datasets()
test_ready = test.concatenate_datasets()

train_ready.save(rsc_dir + "/train_dataset")
test_ready.save(rsc_dir + "/test_dataset")

# All up code is equal to code below:
train_ready, test_ready = ecgdataset.prepare_train_and_test(rsc_dir + "/mit_records", rsc_dir + "/ecgdataset",
                                                            rsc_dir + "/train_dataset",
                                                            rsc_dir + "/test_dataset", test_size=0.25, random_state=42)

