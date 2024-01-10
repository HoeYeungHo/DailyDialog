import torch
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

test_dataloader = get_DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)

model = get_model(path = "last")[0]
model = model.to(device)

model.eval()
first = 1

for output, emotion, act in test_dataloader:
    emotion_hat, act_hat = model(output)

    if first:
        emotionALL = emotion[emotion != -100]
        actALL = act[act != -100]
        EmotionPredictALL = emotion_hat.argmax(dim = 1)[emotion != -100]
        AccPredictALL = act_hat.argmax(dim = 1)[act != -100]
        first = 0
    else:
        emotionALL = torch.cat((emotionALL, emotion[emotion != -100]), dim=0)
        actALL = torch.cat((actALL, act[act != -100]), dim=0)
        EmotionPredictALL = torch.cat((EmotionPredictALL, emotion_hat.argmax(dim = 1)[emotion != -100]), dim=0)
        AccPredictALL = torch.cat((AccPredictALL, act_hat.argmax(dim = 1)[act != -100]), dim=0)

print("ALLACC:{:.4f}, EmotionACC:{:.4f}, EmotionF1Score:{:.4f}, ActACC:{:.4f}, ActF1Score:{:.4f}".format(
    (sum(EmotionPredictALL == emotionALL) / len(emotionALL) + sum(AccPredictALL == actALL) / len(actALL)) / 2,
    sum(EmotionPredictALL == emotionALL) / len(emotionALL),
    f1_score(EmotionPredictALL.cpu(), emotionALL.cpu(), average='macro'),
    sum(AccPredictALL == actALL) / len(actALL),
    f1_score(AccPredictALL.cpu(), actALL.cpu(), average='macro'),
))