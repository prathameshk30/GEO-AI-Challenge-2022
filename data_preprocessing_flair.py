# -*- coding: utf-8 -*-
"""Data_Preprocessing_Flair.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1phA26XjQpKSOQmXrWxAlSg1qSCWtg6ey

#Data Preprocessing
"""

!git clone https://github.com/rsuwaileh/IDRISI.git

"""# Train File"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import shutil
# 
# path = "/content/" + "IDRISI/LMR/data/"
# 
# events = ["california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019", "ecuador_earthquake_2016", 
#           "greece_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017", 
#           "hurricane_irma_2017", "hurricane_maria_2017", "hurricane_matthew_2016", "italy_earthquake_aug_2016", 
#           "kaikoura_earthquake_2016", "kerala_floods_2018", "maryland_floods_2018", "midwestern_us_floods_2019", 
#           "pakistan_earthquake_2019", "puebla_mexico_earthquake_2017", "srilanka_floods_2017"]
# 
# train_path_list = []
# test_path_list = []
# for typ in ['typeless']:
#     for case in ['random']:
#         for event in events:
#             in_path = path + "EN/gold-" + case + "-bilou/" + event 
#             train_path = in_path + "/train.txt"
#             test_path = in_path + "/dev.txt"
#             train_path_list.append(train_path)
#             test_path_list.append(test_path)
# print(train_path_list)
# print(test_path_list)
# 
# 
# with open('TRAIN.txt','wb') as wfd:
#     for f in train_path_list:
#         with open(f,'rb') as fd:
#             shutil.copyfileobj(fd, wfd)

"""# Test/ValidationFile"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import shutil
# 
# path = "/content/" + "IDRISI/LMR/data/"
# 
# events = ["california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019", "ecuador_earthquake_2016", 
#           "greece_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017", 
#           "hurricane_irma_2017", "hurricane_maria_2017", "hurricane_matthew_2016", "italy_earthquake_aug_2016", 
#           "kaikoura_earthquake_2016", "kerala_floods_2018", "maryland_floods_2018", "midwestern_us_floods_2019", 
#           "pakistan_earthquake_2019", "puebla_mexico_earthquake_2017", "srilanka_floods_2017"]
# 
# train_path_list = []
# test_path_list = []
# for typ in ['typeless']:
#     for case in ['random']:
#         for event in events:
#             in_path = path + "EN/gold-" + case + "-bilou/" + event 
#             train_path = in_path + "/train.txt"
#             test_path = in_path + "/dev.txt"
#             train_path_list.append(train_path)
#             test_path_list.append(test_path)
# print(train_path_list)
# print(test_path_list)
# 
# 
# with open('TEST.txt','wb') as wfd:
#     for f in test_path_list:
#         with open(f,'rb') as fd:
#             shutil.copyfileobj(fd, wfd)

"""## Train file 
#### Changing tagging format from "BILOU" to "BIOES"
"""

list1=[]
f = open(r"/content/TRAIN.txt", encoding="utf-8")
for x in f:
    list1.append((x).strip('\n'))
print(list1)

list2=[]
for x in list1:
    if x=='':
        list2.append(x)
    else:
        if x.split()[1]=='O':
            list2.append(x)
        else:
            list2.append(x.replace(x.split()[1],x.split()[1].split('-')[0] + '-' + "LOC"))   

list3=[]
import re
for x in list2:
    x=(re.sub("U-", "S-", str(x)))
    x=(re.sub("L-", "E-", str(x)))
    list3.append(x)

with open(r'/content/TRAIN_final.txt', 'w', encoding="utf-8") as f:
    for line in list3:
        f.write(f"{line}\n")

list1=[]
f = open(r"/content/TEST.txt", encoding="utf-8")
for x in f:
    list1.append((x).strip('\n'))
print(list1)

list2=[]
for x in list1:
    if x=='':
        list2.append(x)
    else:
        if x.split()[1]=='O':
            list2.append(x)
        else:
            list2.append(x.replace(x.split()[1],x.split()[1].split('-')[0] + '-' + "LOC"))   

list3=[]
import re
for x in list2:
    x=(re.sub("U-", "S-", str(x)))
    x=(re.sub("L-", "E-", str(x)))
    list3.append(x)

with open(r'/content/TEST_final.txt', 'w', encoding="utf-8") as f:
    for line in list3:
        f.write(f"{line}\n")

"""# Modeling- Flair"""

!pip install flair

from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0: 'text', 1: 'ner'}
corpus: Corpus = ColumnCorpus('/content/data',columns,
                              train_file='/content/TRAIN_final.txt',
                              dev_file='/content/TEST_final.txt',
                              test_file='/content/TEST_final.txt')

import pandas as pd
data = [[len(corpus.train),  len(corpus.dev)]]
# Prints out the dataset sizes of train test and development in a table.
pd.DataFrame(data, columns=["Train", "Development"])

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
import torch

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

# 4. initialize fine-tuneable transformer embeddings WITH document context
from flair.embeddings import TransformerWordEmbeddings

embeddings = TransformerWordEmbeddings(
    model='tner/deberta-v3-large-ontonotes5',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger = SequenceTagger(
    hidden_size=1024,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=True,
    loss_weights = {'O': 0.0, 'S-LOC': 1.0, 'B-LOC': 1.0, 'E-LOC': 1.0, 'I-LOC': 1.0}
)

# 6. initialize trainer with AdamW optimizer
from flair.trainers import ModelTrainer
from torch.optim import AdamW
trainer = ModelTrainer(tagger, corpus)

# 7. run training with XLM parameters (20 epochs, small LR)
from torch.optim.lr_scheduler import OneCycleLR

trainer.train('ner-english-ontonotes-large',
              learning_rate=5.0e-6,
              mini_batch_size=7,
              mini_batch_chunk_size=1,
              max_epochs=3,
              optimizer=AdamW,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.
              )

import torch
torch.cuda.empty_cache()

import gc
gc.collect()

weights = {}
for key in tagger.label_dictionary.get_items():
  if key == 'O':
    weights[key] = 0.0
  else:
    weights[key] = 1.0
weights

from flair.data import Sentence
from flair.models import SequenceTagger

input_sentence = "In the wake of #HurricaneHarvey's devastation we're taking action. Join us in donating towards relief efforts: at the walmart in College Station tomorrow."
sentence: Sentence = Sentence(input_sentence)
tagger.predict(sentence)
print(sentence.to_tagged_string())

tagger.save('drive/MyDrive/model.pt')

import os
os.listdir()