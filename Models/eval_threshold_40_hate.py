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


test_dataset = pd.read_csv('/raid/home/vibhu20150/temp/Datasets-Processed/H3_Multiclass_Hate_Speech_Detection_test_preprocessed.csv')

print(test_dataset.shape)
print(test_dataset)

from torch.utils.data import Dataset

class CustomTestDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tweets = [tokenizer(tweet, max_length=250, padding='max_length', truncation=True, return_tensors="pt") for tweet in dataset['tweet']]
        self.ids = [id for id in dataset['id']]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        curr_id = self.ids[idx]

        return tweet, curr_id

    def getIds(self):
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
        # self.soft_max = nn.Softmax(dim=1) 

    def forward(self, input_ids, bert_mask):
        seq_last_hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=bert_mask, return_dict=False)
        # print("SEQ SHAPE: ", seq_last_hidden_states.shape) 
        # print("Pool Shape: ", pooled_output.shape)
        pooled_hidden_states = self.pooling_layer(seq_last_hidden_states)
        pooled_hidden_states = pooled_hidden_states.reshape(pooled_output.shape[0], -1)
        linear_output_1 = self.tan_h(self.linear_layer_1(pooled_hidden_states))
        linear_output = self.relu(self.linear_layer(linear_output_1))
        return linear_output



from tqdm import tqdm
from torch.utils.data import DataLoader
# !nvidia-smi

def eval(model, test_data, device):
    model.eval()

    if(device == "cuda"):
        model.cuda()
    batch_size = 16
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    test_preds = []
    sof_max = nn.Softmax(dim=1) 
    with torch.no_grad():
            for batch, (tweets, ids) in enumerate(test_dataloader):
                output = model(tweets["input_ids"].squeeze(1).to(device), tweets["attention_mask"].to(device))
                # print(f"label shape: {labels.shape}") # should be batch X 1
                # loss = loss_function(output, labels)
                # print("output shape: ", output.shape)
                
                # preds = model() 
                # accuracy 
                preds = sof_max(output)

                for i in range(preds.shape[0]): #hate class
                    if preds[i][0] >= 0.4:
                        preds[i][0] = 1

                preds = preds.argmax(dim=1)

                for i in range(len(ids)):
                    # print(i)
                    test_preds.append([preds[i].cpu().item(), ids[i].item()])

    return test_preds


def createCSV(model_preds):
    header = ["label", "id"]

    with open("/raid/home/vibhu20150/temp/predicitions/ninju-preds.csv", 'w', encoding='UTF8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        for row in model_preds:
            csv_writer.writerow(row) 



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.current_device()
torch.cuda.get_device_name(torch.cuda.current_device())

model = torch.load("/raid/home/vibhu20150/temp/BALALALALA-finetune-pooled-bert-2-lin-3-best.pth")
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
