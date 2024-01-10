import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from get_times import get_times, getNum


class DialogueEmotionModel(nn.Module):
    def __init__(self,
                 bert_model,
                 num_emotion,
                 num_act,
                 max_length = 128,
                 device = torch.device("cuda"),
                 double = False
                 ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        if double:
            self.fc_emotion1 = nn.Linear(self.bert_model.config.hidden_size, 512)
            self.fc_emotion2 = nn.Linear(512, num_emotion)
            self.fc_act1 = nn.Linear(self.bert_model.config.hidden_size, 512)
            self.fc_act2 = nn.Linear(512, num_act)
        else:
            self.fc_emotion = nn.Linear(self.bert_model.config.hidden_size, num_emotion)
            self.fc_act = nn.Linear(self.bert_model.config.hidden_size, num_act)
        self.max_length = max_length
        self.device = device
        self.double = double

    def forward(self, output):
        output = self.tokenizer.batch_encode_plus(output, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt').to(self.device)
        output = self.bert_model(**output)[0][:, 0, :]
        if self.double:
            return \
            nn.functional.softmax(self.fc_emotion2(nn.functional.relu(self.fc_emotion1(output))), dim = 1), \
            nn.functional.softmax(self.fc_act2(nn.functional.relu(self.fc_act1(output))), dim = 1)
        else:
            return \
            nn.functional.softmax(self.fc_emotion(output), dim = 1), \
            nn.functional.softmax(self.fc_act(output), dim = 1)


def get_model(bert = "Base", double = False, path = ""):
    if path == "":
        num_emotion = 6
        num_act = 4
        print("Train on the pretrain model")
        if bert == "Large":
            bert_model = "bert-large-uncased"
        else:
            bert_model = "bert-base-uncased"
        return DialogueEmotionModel(bert_model, num_emotion, num_act, double=double), get_times(), -1
    elif path == "last":
        filenameList = os.listdir("model")
        if ".ipynb_checkpoints" in filenameList:
            filenameList.remove(".ipynb_checkpoints")
        filenameList.sort(key = getNum)
        path = "model/" + filenameList[-1]
    print("Load model from {}".format(path))
    return torch.load(path), int(path.split("/")[-1].split("_")[0]), int(path.split("/")[-1].split("_")[-2]) * 0.01
