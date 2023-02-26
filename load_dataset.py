import setting
from collections import Counter
import random

random.seed(1)


def load_dataset(path):
    source = []
    with open(path, 'r', encoding="ISO-8859-1") as reader:  # utf-8 scheme can generate error
        source = reader.readlines()
        source = [line.replace('\n', '') for line in source]
        reader.close()
        return source


# divied each sentence into tags and context
# for every sentence, the content before the first space is combined tag, after is context
def fetch_tags_context(dataset):
    tags = [line.split(' ', 1)[0] for line in dataset]
    context = [line.split(' ', 1)[1] for line in dataset]
    return context, tags


# split combined tags into coarse tags and fine tags
# the parameter should be the combined tag
def divide_tags(tag_set):
    coarse = [line.split(':', 1)[0] for line in tag_set]
    fine = [line.split(':', 1)[1] for line in tag_set]
    return coarse, fine


def load_stopwards(path):
    with open(path, 'r') as reader1:
        stopwords = reader1.readlines()
        stopwords = [i.replace('\n', '') for i in stopwords]
        reader1.close()
        return stopwords


def preprocessing(processing_set, stopwords, lc=False, is_processed=False):  # input is the set need to be preprocessed
    if not is_processed:
        return processing_set
    punc = '''!''()-[]``{}`;:'"\,<>./?@#$%^&*_~'''
    results = []
    for line in processing_set:
        temp = []
        for elements in line.split():
            if elements in stopwords:
                continue
            if elements in punc:
                continue
            else:
                if lc:
                    temp.append(elements.lower())
                else:
                    temp.append(elements)
        results.append(' '.join(temp))

    processing_set = results
    return processing_set


# load source file (training and evaluation) and test file
source = load_dataset(setting.path_train)  # contain training and evaluation set
test_data = load_dataset(setting.path_test)
random.shuffle(source)

# split the source file
# train_test_split: 90% train set, 10% evaluation set
train_len = round(len(source) * 0.9)
train_set = source[:train_len]
dev_set = source[train_len:]

# print(train_set[:2])

# split tags and context for training set, test set and evaluation set
train_X, train_Y = fetch_tags_context(train_set)
dev_X, dev_Y = fetch_tags_context(dev_set)
test_X, test_Y = fetch_tags_context(test_data)

# print(train_X[:2])
# print(train_Y[:2])

# divide tags into coarse and fine tags
# until below each element in train_X correspond to two tags
# each of them is from: train_coarse and train_fine
train_coarse, train_fine = divide_tags(train_Y)
dev_coarse, dev_fine = divide_tags(dev_Y)
test_coarse, test_fine = divide_tags(test_Y)

# print(train_coarse[:2])
# print(train_fine[:2])

# match classes to encoded number
lab_coarse = list(set(train_coarse))
lab_fine = list(set(train_fine))

# print(lab_coarse)
# print(lab_fine)
lab_dict1 = {lab_coarse[i]: i for i in range(len(lab_coarse))}
lab_dict2 = {lab_fine[i]: i for i in range(len(lab_fine))}

# print(lab_dict1)
# print(lab_dict2)

# number of classes
num_coarse = len(lab_coarse)
num_fine = len(lab_fine)

# load stopwords
stopword_file = load_stopwards(setting.path_stop)

# preprocessed training set
train_X = preprocessing(processing_set=train_X, stopwords=stopword_file, lc=True, is_processed=True)
test_X = preprocessing(processing_set=test_X, stopwords=stopword_file, lc=True, is_processed=True)
dev_X = preprocessing(processing_set=dev_X, stopwords=stopword_file, lc=True, is_processed=True)


