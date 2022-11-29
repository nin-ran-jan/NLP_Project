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

# dataset = np.array(pd.read_csv('/content/drive/MyDrive/NLP_Project_sem5/processed_data.csv'))
# dataset = pd.read_csv('/content/drive/MyDrive/NLP_Project_sem5/processed_data.csv')

train_dataset = pd.read_csv('/ssd-scratch/vibhu20150/temp/Datasets-Processed/H3_Multiclass_Hate_Speech_Detection_train_preprocessed.csv')
test_dataset = pd.read_csv('/ssd-scratch/vibhu20150/temp/Datasets-Processed/H3_Multiclass_Hate_Speech_Detection_test_preprocessed.csv')

print(train_dataset)
print(test_dataset)

print(train_dataset.shape)
# print(dataset[0,2])
print(train_dataset)
print(test_dataset.shape)
# print(dataset[0,2])
print(test_dataset)

# num_labels = 3
# tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

# model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", problem_type="multi_label_classification", )

# print(model.config.id2label)

# inputs = tokenizer(dataset['tweet'][0], return_tensors="pt")
# print(inputs.items())
# print(inputs['input_ids'].squeeze(1).shape)
# print(tokenizer.decode(inputs["input_ids"][0]))
# logits = model(**inputs).logits
# print(logits)
# predicted_class_id = logits.argmax().item()
# print(predicted_class_id)
# model.config.id2label[predicted_class_id]

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, class_type="all"):
        if class_type == "all":
            self.labels = [label for label in dataset['label']]
            self.tweets = [tokenizer(tweet, max_length=512, padding='max_length', truncation=True, return_tensors="pt") for tweet in dataset['tweet']]
        elif class_type == "hate":
            self.labels = [label for label in dataset['label'] if label == 0]
            self.tweets = [tokenizer(row[1], max_length=512, padding='max_length', truncation=True, return_tensors="pt") for row in np.array(dataset) if row[0] == 0] 


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]

        return tweet, label

    def getLabels(self):
        return self.labels

# sequence classification/regression head on top a linear layer on top of the pooled output of BERT
from transformers import BertModel


class BertSeqPoolClassifier(nn.Module):
    def __init__(self):
        super(BertSeqPoolClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.LazyLinear(3)
        self.pooling_layer = nn.AvgPool1d(384, stride=192)
        self.relu = nn.ReLU()
        # self.soft_max = nn.Softmax(dim=3) 

    def forward(self, input_ids, bert_mask):
        seq_last_hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=bert_mask, return_dict=False)
        # print("SEQ SHAPE: ", seq_last_hidden_states.shape) 
        # print("Pool Shape: ", pooled_output.shape)
        pooled_hidden_states = self.pooling_layer(seq_last_hidden_states)
        pooled_hidden_states = pooled_hidden_states.reshape(pooled_output.shape[0], -1)
        linear_output = self.relu(self.linear_layer(pooled_hidden_states))
        # probs = self.soft_max(linear_output) # already incorporated in crossentropyLoss
        # return probs
        return linear_output

from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model, train_data, test_data, learning_rate_bert, learning_rate_lin, epochs, device):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer_bert = torch.optim.Adam(model.bert.parameters(), lr=learning_rate_bert)
    optimizer_lin = torch.optim.Adam(model.linear_layer.parameters(), lr=learning_rate_lin)
    if(device == "cuda"):
        model.cuda()
        loss_function.cuda()

    
    batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    best_score = 0.769
    
    for epoch in tqdm(range(epochs)):

        correct_preds = 0
        train_preds = []
        test_preds = []

        for batch, (tweets, labels) in enumerate(train_dataloader):
            
            labels = labels.to(device)
            # print()
            # print("="*40)
            # print()
            # print(tweets["attention_mask"].shape)
            # print()
            output = model(tweets["input_ids"].squeeze(1).to(device), tweets["attention_mask"].to(device))
            # print(f"label shape: {labels.shape}") # should be batch X 1
            loss = loss_function(output, labels)
            # print("output shape: ", output.shape)
            
            # preds = model() 

            # update
            optimizer_bert.zero_grad()
            optimizer_lin.zero_grad()
            loss.backward()
            optimizer_bert.step()
            optimizer_lin.step()

            # accuracy 
            preds = output.argmax(dim=1)
            # print("preds shape: ", preds.shape)
            # print("labels shape: ", len(labels))

            for i in range(len(labels)):
                # print(i)
                train_preds.append(preds[i].cpu())
                if preds[i] == labels[i]:
                    correct_preds += 1
            
        
        


        # testing code 

        test_correct_preds = 0

        with torch.no_grad():
            for batch, (tweets, labels) in enumerate(test_dataloader):
                labels = labels.to(device)
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
                # print("preds shape: ", preds.shape)
                # print("labels shape: ", labels.shape)

                for i in range(len(labels)):
                    # print(i)
                    test_preds.append(preds[i].cpu())
                    if preds[i] == labels[i]:
                        test_correct_preds += 1
        

          
        print(f"\nEpoch: {epoch + 1}, Train Acc: {correct_preds/len(train_data)}, Train Length: {len(train_data)}, Test Acc: {test_correct_preds/len(test_data)}, Test Length: {len(test_data)}")
        print(f"F1 Score Train: {f1_score(train_data.getLabels(), train_preds, average=None)}")
        print(f"Macro F1 Score Train: {f1_score(train_data.getLabels(), train_preds, average='macro')}")
        print(f"F1 Score Test: {f1_score(test_data.getLabels(), test_preds, average=None)}")
        print(f"Macro F1 Score Test: {f1_score(test_data.getLabels(), test_preds, average='macro')}")
        curr_score = f1_score(test_data.getLabels(), test_preds, average='macro') 

        if curr_score > best_score:
            best_score = curr_score
            torch.save(model, "./pooled-bert-3-best.pth")
        # save best model


# !nvidia-smi

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.current_device()
torch.cuda.get_device_name(torch.cuda.current_device())
# print()
# for row in np.array(train_dataset)[:1]:
#     print(row.shape)
#     print(row[0], row[1])
#     exit()
    # print(row['tweet'])

# model = BertSeqPoolClassifier()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = torch.load("/ssd-scratch/vibhu20150/temp/pooled-bert-3-best.pth")

print(len([para for para in model.parameters()]))
print(len([para for para in model.bert.parameters()]))
print(len([para for para in model.linear_layer.parameters()]))

labels_map = {0: "hate?",
              1: "offensive?",
              2: "none?",
              }

# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# print(target.shape)
# print(torch.device("cuda"))

# print(dataset)
# print(tokenizer)
# train_data = CustomDataset(train_dataset, tokenizer)



train_data = CustomDataset(train_dataset[:15860], tokenizer) 
test_data = CustomDataset(train_dataset[15860:], tokenizer)

print("TRAIN LENGTH: ", len(train_data))

train(model, train_data, test_data, 0.00001, 0.001, 3, device)


# torch.save(model.state_dict(), "./trained-model-states.pth")
# torch.save(model, "./pooled-bert-3.pth")