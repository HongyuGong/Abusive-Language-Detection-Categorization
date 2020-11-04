import os

# path
data_folder = os.path.join(os.path.dirname(os.getcwd()), "data")
dump_folder = os.path.join(os.path.dirname(os.getcwd()), "dump/")
classification_dataset = os.path.join(data_folder, "Anonymized_Sentences_Classified.csv")
categorization_dataset = os.path.join(data_folder, "Anonymized_Comments_Categorized.csv")

# hyperparameter
max_sent_len = 100
#unk = "<UNK>"
unk = "<PAD/>"
pad = "<PAD/>"


