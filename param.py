import os

# path
data_folder = "data/"
dump_folder = "dump/"
classification_dataset = os.path.join(data_folder, "Anonymized_Sentences_Classified.csv")
categorization_dataset = os.path.join(data_folder, "Anonymized_Comments_Categorized.csv")

# hyperparameter
max_sent_len = 300
unk = "<UNK>"
pad = "<PAD>"


