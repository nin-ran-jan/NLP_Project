# -*- coding: utf-8 -*-
"""BERT_CLASSIFIER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13s_yhJi2rvqc2WZ9ffpZjc20gXY3zU3d
"""

# using bert sequence classifier from hugging face 
# https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/bert#transformers.BertForSequenceClassification
# !pip install transformers

# from google.colab import drive
# drive.mount('/content/drive')

import torch
from transformers import BertTokenizer, BertForSequenceClassification 
import pandas as pd
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import f1_score
import csv


# dataset = np.array(pd.read_csv('/content/drive/MyDrive/NLP_Project_sem5/processed_data.csv'))
# dataset = pd.read_csv('/content/drive/MyDrive/NLP_Project_sem5/processed_data.csv')

# train_dataset = pd.read_csv('/ssd-scratch/vibhu20150/temp/Datasets-Processed/H3_Multiclass_Hate_Speech_Detection_train_preprocessed.csv')
test_dataset = pd.read_csv('/ssd-scratch/vibhu20150/temp/Datasets-Processed/H3_Multiclass_Hate_Speech_Detection_test_preprocessed.csv')

# print(train_dataset)
# print(test_dataset)

# print(train_dataset.shape)
# # print(dataset[0,2])
# print(train_dataset)
print(test_dataset.shape)
# print(dataset[0,2])
print(test_dataset)

from torch.utils.data import Dataset

class CustomTestDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        # self.labels = [label for label in dataset['label']]
        self.tweets = [tokenizer(tweet, max_length=512, padding='max_length', truncation=True, return_tensors="pt") for tweet in dataset['tweet']]
        self.ids = [id for id in dataset['id']]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        curr_id = self.ids[idx]

        return tweet, curr_id

    def getIds():
        return self.ids



# sequence classification/regression head on top a linear layer on top of the pooled output of BERT
from transformers import BertModel


class BertSeqPoolLinClassifier(nn.Module):
    def __init__(self):
        super(BertSeqPoolLinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear_layer_1 = nn.LazyLinear(100)
        self.linear_layer = nn.LazyLinear(3)

        self.pooling_layer = nn.AvgPool1d(384, stride=192)
        self.relu = nn.ReLU()
        self.tan_h = nn.Tanh()
        # self.soft_max = nn.Softmax(dim=3) 

    def forward(self, input_ids, bert_mask):
        seq_last_hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=bert_mask, return_dict=False)
        # print("SEQ SHAPE: ", seq_last_hidden_states.shape) 
        # print("Pool Shape: ", pooled_output.shape)
        pooled_hidden_states = self.pooling_layer(seq_last_hidden_states)
        pooled_hidden_states = pooled_hidden_states.reshape(pooled_output.shape[0], -1)
        linear_output_1 = self.tan_h(self.linear_layer_1(pooled_hidden_states))
        linear_output = self.relu(self.linear_layer(linear_output_1))
        # probs = self.soft_max(linear_output) # already incorporated in crossentropyLoss
        # return probs
        return linear_output



from tqdm import tqdm
from torch.utils.data import DataLoader
# !nvidia-smi

def eval(model, test_data, device):
    model.eval()

    if(device == "cuda"):
        model.cuda()
    batch_size = 8
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    test_preds = []

    with torch.no_grad():
            for batch, (tweets, ids) in enumerate(test_dataloader):
                # print()
                # print("="*40)
                # print()
                # print(tweets["attention_mask"].shape)
                # print()
                output = model(tweets["input_ids"].squeeze(1).to(device), tweets["attention_mask"].to(device))
                # print(f"label shape: {labels.shape}") # should be batch X 1
                # loss = loss_function(output, labels)
                # print("output shape: ", output.shape)
                
                # preds = model() 
                # accuracy 
                preds = output.argmax(dim=1)

                for i in range(len(ids)):
                    # print(i)
                    test_preds.append([preds[i].cpu().item(), ids[i].item()])

    return test_preds


def createCSV(model_preds):
    header = ["label", "id"]

    with open("/ssd-scratch/vibhu20150/temp/predictions/preds.csv", 'w', encoding='UTF8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        for row in model_preds:
            csv_writer.writerow(row) 



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.current_device()
torch.cuda.get_device_name(torch.cuda.current_device())

model = torch.load("/ssd-scratch/vibhu20150/temp/pooled-bert-lin-3-best.pth")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(len([para for para in model.parameters()]))
print(len([para for para in model.bert.parameters()]))
print(len([para for para in model.linear_layer.parameters()]))

labels_map = {0: "hate?",
              1: "offensive?",
              2: "none?",
              }


test_data = CustomTestDataset(test_dataset, tokenizer)
print("test dataset: ", test_dataset['id'].shape)
model_preds = eval(model, test_data, device)
createCSV(model_preds)
