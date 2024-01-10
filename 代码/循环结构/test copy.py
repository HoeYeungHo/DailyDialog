import torch
from torch import nn
from DataProcessing import *
from Model import get_model
from sklearn.metrics import f1_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

test_dataset = DailyDialog(
    TextFile="test/dialogues_test.txt", 
    EmotionFile="test/dialogues_emotion_test.txt",
    ActFile="test/dialogues_act_test.txt",
    device=device
    )

batch_size = 8  # 批量大小

test_dataloader = get_DataLoader(dataset = test_dataset, batch_size=batch_size)


model = get_model("last")[0]
model = model.to(device)

model.eval()
first = 1

for output, emotion, act, SentenceNum, WordNum, batch_size in test_dataloader:
    emotion_hat, act_hat = model(output, SentenceNum, batch_size)

    if first:
        emotionALL = emotion[emotion != -100]
        actALL = act[act != -100]
        EmotionPredictALL = emotion_hat.flatten(0, 1)[emotion != -100]
        AccPredictALL = act_hat.flatten(0, 1)[act != -100]
        first = 0
    else:
        emotionALL = torch.cat((emotionALL, emotion[emotion != -100]), dim=0)
        actALL = torch.cat((actALL, act[act != -100]), dim=0)
        EmotionPredictALL = torch.cat((EmotionPredictALL, emotion_hat.flatten(0, 1)[emotion != -100]), dim=0)
        AccPredictALL = torch.cat((AccPredictALL, act_hat.flatten(0, 1)[act != -100]), dim=0)

EmotionPredictLable = EmotionPredictALL.argmax(dim = 1)
ActPredictLable = AccPredictALL.argmax(dim = 1)

print("EmotionACC:{:.4f}, EmotionF1Score:{:.4f}, EmotionLoss:{:.4f}, ActACC:{:.4f}, ActF1Score:{:.4f}, ActLoss:{:.4f}, Loss:{:.4f}".format(
    sum(EmotionPredictLable == emotionALL) / len(emotionALL),  # EmotionACC
    f1_score(EmotionPredictLable.cpu(), emotionALL.cpu(), average='micro'),  # EmotionF1Score
    nn.functional.cross_entropy(EmotionPredictALL, emotionALL),  # EmotionLoss
    sum(ActPredictLable == actALL) / len(actALL),  # ActACC
    f1_score(ActPredictLable.cpu(), actALL.cpu(), average='micro'),  # ActF1Score
    nn.functional.cross_entropy(AccPredictALL, actALL),  # ActLoss
    nn.functional.cross_entropy(EmotionPredictALL, emotionALL) + nn.functional.cross_entropy(AccPredictALL, actALL) # Loss
))