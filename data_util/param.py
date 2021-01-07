import os

# path
data_folder = os.path.join(os.path.dirname(os.getcwd()), "data")
dump_folder = os.path.join(os.path.dirname(os.getcwd()), "dump/")

stormfront_data_folder = os.path.join(os.path.dirname(os.getcwd()), "stormfront_data/")
stormfront_classification_dataset = os.path.join(stormfront_data_folder, "hate_speech_classified_new2.csv")

classification_dataset = os.path.join(data_folder, "Anonymized_Sentences_Classified.csv")
categorization_dataset = os.path.join(data_folder, "Anonymized_Comments_Categorized.csv")

# hyperparameter
max_sent_len = 100
unk = "<UNK>"
pad = "<PAD/>"
emb_dim = 300
