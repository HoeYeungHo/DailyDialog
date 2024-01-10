import os
import pandas as pd

data = pd.read_excel("结果记录表.xlsx")

for i in range(len(data)):
    os.rename(data["登记表名"][i] + ".csv", data["文件名"][i] + ".csv")