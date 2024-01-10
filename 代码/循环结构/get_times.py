import os

def getNum(filename):
    sum = ''
    for char in filename:
        if char in "0123456789":
           sum += char
        else:
            return int(sum)

    
def get_times(path = "model"):
    filenameList = os.listdir(path)
    if ".ipynb_checkpoints" in filenameList:
        filenameList.remove(".ipynb_checkpoints")
    if len(filenameList) == 0: return 0
    else:
        filenameList.sort(key = getNum)
        return int(filenameList[-1].strip(":")[0]) + 1