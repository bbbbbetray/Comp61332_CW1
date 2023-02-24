import configparser

config = configparser.ConfigParser()
config.sections()
config.read("config.ini")
# print(config.keys())

# define path files
path_train = config["PATH"]["path_train"]
path_dev = config["PATH"]["path_dev"]
path_test = config["PATH"]["path_test"]
path_stop = config["PATH"]['stop_words']

# define hyper parameters
embedding_dim = int(config['HYPER_PARAMETERS']['word_embedding_dim'])
batch_size = int(config['HYPER_PARAMETERS']['batch_size'])
learning_rate = float(config['HYPER_PARAMETERS']['learning_rate'])
hidden_size = int(config['HYPER_PARAMETERS']['hidden_size'])
num_epochs = int(config['MODEL_SETTING']['epoch'])








