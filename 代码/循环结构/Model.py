import os
import torch
import torch.nn as nn
from transformers import BertModel
from get_times import get_times, getNum


class DialogueEmotionModel(nn.Module):
    def __init__(self,
                 bert_model,
                 lstm_hidden_size,
                 lstm_num_layers,
                 num_emotion=6,
                 num_act=4,
                 bidirectional = False,
                 dropout=0.0,
                 RNN="LSTM",
                 double = False
                 ):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model)
        if RNN == "LSTM":
            self.rnn = nn.LSTM(input_size=self.bert_model.config.hidden_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=lstm_num_layers,
                                batch_first=True,
                                bidirectional=bidirectional,
                                dropout=dropout
                                )
        elif RNN == "RNN":
            self.rnn = nn.RNN(input_size=self.bert_model.config.hidden_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=lstm_num_layers,
                                batch_first=True,
                                bidirectional=bidirectional,
                                dropout=dropout
                                )
        elif RNN == "GRU":
            self.rnn = nn.GRU(input_size=self.bert_model.config.hidden_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=lstm_num_layers,
                                batch_first=True,
                                bidirectional=bidirectional,
                                dropout=dropout
                                )
        linear_input_size = 2 * lstm_hidden_size if bidirectional else lstm_hidden_size
        self.double = double
        if double:
            self.fc_emotion1 = nn.Linear(linear_input_size, 1024)
            self.dropout_emotion = torch.nn.Dropout(p = dropout, inplace = True)
            self.fc_emotion2 = nn.Linear(1024, num_emotion)
            self.fc_act1 = nn.Linear(linear_input_size, 1024)
            self.dropout_act = torch.nn.Dropout(p = dropout, inplace = True)
            self.fc_act2 = nn.Linear(1024, num_act)
        else:
            self.fc_emotion = nn.Linear(linear_input_size, num_emotion)
            self.fc_act = nn.Linear(linear_input_size, num_act)

    def forward(self, output, SentenceNum, batch_size):
        output = self.bert_model(**output)[0][:, 0, :]
        output = output.reshape(batch_size, SentenceNum, -1)
        output, _ = self.rnn(output)
        output = nn.functional.relu(output)

        if self.double:
            return \
            nn.functional.softmax(self.fc_emotion2(self.dropout_emotion(nn.functional.relu(self.fc_emotion1(output)))), dim = 1), \
            nn.functional.softmax(self.fc_act2(self.dropout_act(nn.functional.relu(self.fc_act1(output)))), dim = 1)
        else:
            return \
            nn.functional.softmax(self.fc_emotion(output), dim = 1), \
            nn.functional.softmax(self.fc_act(output), dim = 1)


def get_model(lstm_hidden_size = 1024,
              lstm_num_layers = 2,
              bidirectional = False,
              dropout = 0.0,
              RNN="LSTM",
              double=False,
              bert="Base",
              path = ""):
    if path == "":
        num_emotion = 6
        num_act = 4
        print("Train on the pretrain model")
        if bert == "Large":
            bert_model = "bert-large-uncased"
        else:
            bert_model = "bert-base-uncased"
        return DialogueEmotionModel(bert_model, lstm_hidden_size, lstm_num_layers, num_emotion, num_act, bidirectional, dropout, RNN, double), get_times(), -1
    elif path == "last":
        filenameList = os.listdir("model")
        if ".ipynb_checkpoints" in filenameList:
            filenameList.remove(".ipynb_checkpoints")
        filenameList.sort(key = getNum)
        path = "model/" + filenameList[-1]
    print("Load model from {}".format(path))
    return torch.load(path), int(path.split("/")[-1].split("_")[0]), int(path.split("/")[-1].split("_")[-2]) * 0.01


# 重置BERT模型的嵌入层、编码器层和pooler层的参数
def reset_bert_parameters(model):
    # 重置嵌入层参数
    for param in model.embeddings.parameters():
        nn.init.normal_(param, mean=0, std=0.02)  # 假设使用正态分布初始化

    # 重置编码器层参数
    for layer in model.encoder.layer:
        for param in layer.parameters():
            param.data.normal_(mean=0.0, std=0.02)  # 假设使用正态分布初始化

    # 重置pooler层参数
    model.pooler.apply(reset_linear)

# 重置线性层的参数
def reset_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=0.02)
        nn.init.zeros_(layer.bias)