import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DailyDialog(Dataset):
    def __init__(self,
                 TextFile="dialogues_text.txt",
                 EmotionFile="dialogues_emotion.txt",
                 ActFile="dialogues_act.txt",
                 bert_model="bert-base-uncased",
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
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length
        self.max_sentence = max_sentence
        self.device = device

    def __len__(self):
        return len(self.TextList)

    def __getitem__(self, idx):

        Dialogues = [text.strip()
                     for text in self.TextList[idx].strip().split("__eou__") if len(text) > 0][:self.max_sentence]
        input_ids, token_type_ids, attention_mask = self.tokenizer.batch_encode_plus(Dialogues, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt').to(self.device).values()
        # print(input_ids.shape)
        # columns_to_keep = torch.any(input_ids != 0, dim=0)
        # input_ids = input_ids[:, columns_to_keep]
        # token_type_ids = token_type_ids[:, columns_to_keep]
        # attention_mask = attention_mask[:, columns_to_keep]
        
        sentenceNum, wordNum = input_ids.shape

        Emotion = list(map(int, self.EmotionList[idx].strip().split()))[:self.max_sentence]
        Emotion = torch.tensor(Emotion).to(self.device)
        Emotion = torch.where(Emotion == 0, -99, Emotion)
        Emotion -= 1

        Act = list(map(int, self.ActList[idx].strip().split()))[:self.max_sentence]
        Act = torch.tensor(Act).to(self.device)
        Act -= 1

        return {"input_ids":input_ids, "token_type_ids":token_type_ids, "attention_mask":attention_mask}, sentenceNum, wordNum, Emotion, Act


def custom_collate_fn(batchs, device = torch.device("cuda")):
    SentenceNum, WordNum = np.max([[batch[1], batch[2]] for batch in batchs], axis=0)
    first = 1

    token_type_ids = torch.zeros(SentenceNum * len(batchs), WordNum).to(device)

    for batch in batchs:
        pad_height = max(0, SentenceNum - batch[0]["input_ids"].shape[0])
        pad_width = max(0, WordNum - batch[0]["input_ids"].shape[1])
        if first:
            input_ids = torch.nn.functional.pad(batch[0]["input_ids"], (0, pad_width, 0, pad_height), value=0)
            attention_mask = torch.nn.functional.pad(batch[0]["attention_mask"], (0, pad_width, 0, pad_height), value=0)
            Emotion = torch.nn.functional.pad(batch[3], (0, pad_height), value=-100)
            Act = torch.nn.functional.pad(batch[4], (0, pad_height), value=-100)
            first = 0
        else:
            input_ids = torch.cat((input_ids, torch.nn.functional.pad(batch[0]["input_ids"], (0, pad_width, 0, pad_height), value=0)), dim=0)
            attention_mask = torch.cat((attention_mask, torch.nn.functional.pad(batch[0]["attention_mask"], (0, pad_width, 0, pad_height), value=0)), dim=0)
            Emotion = torch.cat((Emotion, torch.nn.functional.pad(batch[3], (0, pad_height), value=-100)), dim = -1)
            Act = torch.cat((Act, torch.nn.functional.pad(batch[4], (0, pad_height), value=-100)), dim = -1)

    return {"input_ids":input_ids.to(dtype=torch.long), "token_type_ids":token_type_ids.to(dtype=torch.long), "attention_mask":attention_mask.to(dtype=torch.long)}, Emotion, Act, SentenceNum, WordNum, len(batchs)

def get_DataLoader(dataset, batch_size = 8, shuffle=True, collate_fn=custom_collate_fn):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def LoadTrainData(batch_size1 = 4, batch_size2 = 8, device = torch.device("cuda"), bert = "Base"):
    if bert == "Large":
        bert_model = "bert-large-uncased"
    else:
        bert_model = "bert-base-uncased"
    train_dataset = DailyDialog(
        TextFile="train/dialogues_train.txt", 
        EmotionFile="train/dialogues_emotion_train.txt",
        ActFile="train/dialogues_act_train.txt",
        device=device,
        bert_model=bert_model
    )
    valid_dataset = DailyDialog(
        TextFile="validation/dialogues_validation.txt", 
        EmotionFile="validation/dialogues_emotion_validation.txt",
        ActFile="validation/dialogues_act_validation.txt",
        device=device,
        bert_model=bert_model
        )
    
    return get_DataLoader(dataset = train_dataset, batch_size=batch_size1), get_DataLoader(dataset = valid_dataset, batch_size=batch_size2)

