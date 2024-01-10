import os
import torch
from DataProcessing import *
from Model import get_model
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, SequentialLR
from torch import nn
from sklearn.metrics import f1_score
import time

## 参数设置
# Title
title = "Bert_None_预训练_全参_" + str(int(time.time()))
print(title)
# 模型参数
ALLParameter=True
double = False
bert = "Base"
# 训练参数
learn_rate = 1e-6
# Loss比例
alpha = 0.5


if not os.path.exists("model"): os.mkdir("model")
if not os.path.exists("record"): os.mkdir("record")

# 记录文件
file = open("record/{}.csv".format(title), "w")
file.write("epoch,batch,batchALL,ALLACC,EmotionACC,EmotionF1Score,ActACC,ActF1Score\n")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

train_dataloader, valid_dataloader = LoadTrainData(batch_size1 = 8, batch_size2 = 8, device = device)

# 设置模型
model, times, bestEmotionACC = get_model(bert = bert, double=double)
model = model.to(device)

# 参数冻结
if not ALLParameter:  # 冻结参数
    print("冻结BERT")
    for name, param in model.named_parameters():
        if name.startswith('bert_model'):
            param.requires_grad = False


# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

milestones1 = 10
milestones2 = 200
warmup_scheduler = ConstantLR(optimizer, total_iters=milestones1)
decay_scheduler = ExponentialLR(optimizer, gamma=0.95)
combined_scheduler = SequentialLR(optimizer,
                                  [warmup_scheduler, decay_scheduler],
                                  milestones=[milestones2])

batchALL = 0
for epoch in range(1):
    batch = 0
    for output, emotion, act in train_dataloader:
        # 训练阶段
        model.train()
        optimizer.zero_grad()  # 梯度清零
        emotion_hat, act_hat = model(output)

        loss = nn.functional.cross_entropy(emotion_hat, emotion)
        if torch.isnan(loss):
            loss = alpha * nn.functional.cross_entropy(act_hat, act)
        else:
            loss += alpha * nn.functional.cross_entropy(act_hat, act)
        
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化
        combined_scheduler.step()  # 更新学习率
        
        # 模型评估
        if batch % 100 == 0 or batch < 50 or (batch < 200 and batch % 5 == 0):
            model.eval()
            first = 1
            for output, emotion, act in valid_dataloader:
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
            
            EmotionACC = sum(EmotionPredictALL == emotionALL) / len(emotionALL)
            EmotionF1Score = f1_score(EmotionPredictALL.cpu(), emotionALL.cpu(), average='macro')
            ActACC = sum(AccPredictALL == actALL) / len(actALL)
            ActF1Score = f1_score(AccPredictALL.cpu(), actALL.cpu(), average='macro')
            ALLACC = (EmotionACC + ActACC) / 2
            file.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch, batch, batchALL, ALLACC, EmotionACC, EmotionF1Score, ActACC, ActF1Score))
            print("epoch:{}, batch:{}, batchALL:{}, ALLACC:{:.4f}, EmotionACC:{:.4f}, EmotionF1Score:{:.4f}, ActACC:{:.4f}, ActF1Score:{:.4f}".format(epoch, batch, batchALL, ALLACC, EmotionACC, EmotionF1Score, ActACC, ActF1Score))

            if EmotionACC > bestEmotionACC:
                bestEmotionACC = EmotionACC
                if EmotionACC > 0.6:
                    torch.save(model, "model/{}_BERT_{}_{}".format(times, int(EmotionACC*100), int(time.time())))  # 保存模型
                    print("Save the model model/{}_BERT_{}_{}".format(times, int(EmotionACC*100), int(time.time())))
            
        batch += 1
        batchALL += 1

file.close()

exec(open('test.py').read())