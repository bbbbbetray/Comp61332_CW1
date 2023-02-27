import load_dataset
import setting
from torch.utils.data import DataLoader, Dataset
import utils


# preprocessed training set, need preprocess when use random initialized word embedding
train_X = load_dataset.preprocessing(processing_set=load_dataset.train_X,
                                     stopword_path=setting.path_stop, lc=True, is_processed=True)
test_X = load_dataset.preprocessing(processing_set=load_dataset.test_X,
                                    stopword_path=setting.path_stop, lc=True, is_processed=True)
dev_X = load_dataset.preprocessing(processing_set=load_dataset.dev_X,
                                   stopword_path=setting.path_stop, lc=True, is_processed=True)

# build up vocabulary, words appear more than k times will be selected
vocab = utils.build_vocab(train_X=train_X, min_freq=1)

# get vocab size
vocab_size = len(vocab)

# convert dataset, tagset into dataloader
def to_dataloader(dataset, padding_criteria, TagSet):
    # convert each token to it's corresponding index in the vocabulary
    idx = [[vocab.get(token) if token in vocab else 0 for token in sent.split()] for sent in dataset]

    # pad each sequence to the given length with padding_value: vocab_size
    padded_idx = utils.pad_or_truncate(idx, padding_criteria, vocab_size)

    # manually build dataset for dataloader
    dataset = utils.TextDataset(padded_idx, TagSet)

    # construct dataloader
    data_loader = DataLoader(dataset, batch_size=setting.batch_size, shuffle=True)
    return data_loader


train_loader = to_dataloader(train_X, 25, load_dataset.fine_train)
dev_loader = to_dataloader(dev_X, 25, load_dataset.fine_dev)
test_loader = to_dataloader(test_X, 25, load_dataset.fine_test)




