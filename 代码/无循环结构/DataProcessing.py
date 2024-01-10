import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DailyDialog(Dataset):
    def __init__(self,
                 TextFile="dialogues_text.txt",
                 EmotionFile="dialogues_emotion.txt",
                 ActFile="dialogues_act.txt",
                 max_length = 128,
                 max_sentence = 12,
                 device = torch.device("cpu")
                 ):
        with open(TextFile, "r", encoding="utf-8") as file:
            self.TextList = file.readlines()
        with open(EmotionFile, "r", encoding="utf-8") as file:
            self.EmotionList = file.readlines()
        with open(ActFile, "r", encoding="utf-8") as file:
            self.ActList = file.readlines()
        self.max_length = max_length
        self.max_sentence = max_sentence
        self.device = device

    def __len__(self):
        return len(self.TextList)

    def __getitem__(self, idx):

        Dialogues = [text.strip()
                     for text in self.TextList[idx].strip().split("__eou__") if len(text) > 0][:self.max_sentence]
        
        Emotion = list(map(int, self.EmotionList[idx].strip().split()))[:self.max_sentence]
        Emotion = torch.tensor(Emotion).to(self.device)
        Emotion = torch.where(Emotion == 0, -99, Emotion)
        Emotion -= 1

        Act = list(map(int, self.ActList[idx].strip().split()))[:self.max_sentence]
        Act = torch.tensor(Act).to(self.device)
        Act -= 1


        return Dialogues, Emotion, Act

def custom_collate_fn(batchs):
    return sum([batch[0] for batch in batchs], []), torch.cat([batch[1] for batch in batchs], dim=0), torch.cat([batch[2] for batch in batchs], dim=0)

def get_DataLoader(dataset, batch_size = 8, shuffle=True, collate_fn=custom_collate_fn):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def LoadTrainData(batch_size1 = 4, batch_size2 = 8, device = torch.device("cuda")):
    train_dataset = DailyDialog(
        TextFile="train/dialogues_train.txt", 
        EmotionFile="train/dialogues_emotion_train.txt",
        ActFile="train/dialogues_act_train.txt",
        device=device
    )
    valid_dataset = DailyDialog(
        TextFile="validation/dialogues_validation.txt", 
        EmotionFile="validation/dialogues_emotion_validation.txt",
        ActFile="validation/dialogues_act_validation.txt",
        device=device
        )
    
    return get_DataLoader(dataset = train_dataset, batch_size=batch_size1), get_DataLoader(dataset = valid_dataset, batch_size=batch_size2)

