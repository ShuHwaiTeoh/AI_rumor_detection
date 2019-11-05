import sys
# import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import numpy as np

#X: time, Y:acc
#X: time, Y:epoch
resultPath_torch164DAM = '../result/torch16-4-0.001ADAM.txt'
resultPath_torch164SGD = '../result/torch16-4-0.001SGD.txt'
resultPath_theano164SGD = '../result/theano16-4-0.001SGD.txt'
resultPath_torch154DAM = '../result/torch15-4-0.001ADAM.txt'
resultPath_torch152SGD = '../result/torch15-2-0.005SGD.txt'
resultPath_theano152SGD = '../result/theano15-2-0.005SGD.txt'

def loadResult(fileName):
    timeLine = []
    acc = []
    epoch =[]
    lineNumber = 0
    with open(fileName, "r") as rFile :
        lines = rFile.readlines()
    for line in lines:
        if lineNumber%2 == 0:
            timeLine.append(lines[lineNumber].split(": ")[0].strip())
            epoch.append(lines[lineNumber].split(" ")[-1].strip())
        else:
            acc.append(float(lines[lineNumber].split(",")[1].strip()))
    return timeLine, acc, epoch

def plotResult(timeLine_adam, acc_adam, epoch_adam, timeLine_sgd, acc_sgd, epoch_sgd, timeLine_theano, acc_theano, epoch_theano):
    timeDiff_adam = []
    timeDiff_torchSGD = []
    timeDiff_theanoSGD = []
    adamD1 = datetime.datetime.strptime(timeLine_adam[0], '%Y-%m-%d %H:%M:%S')
    torchSGDD1 = datetime.datetime.strptime(timeLine_sgd[0], '%Y-%m-%d %H:%M:%S')
    theanoSGDD1 = datetime.datetime.strptime(timeLine_theano[0], '%Y-%m-%d %H:%M:%S')
    timeDiff = [timeDiff_adam, timeDiff_torchSGD, timeDiff_theanoSGD]
    timeLine = [timeLine_adam, timeLine_sgd, timeLine_theano]
    timeStart = [adamD1, torchSGDD1, theanoSGDD1]
    for idx,t in enumerate(timeLine):
        for j in t:
            d2 = datetime.datetime.strptime(t[j], '%Y-%m-%d %H:%M:%S')
            timeDiff[idx].append(round((d2-timeStart[idx]).seconds/60,1))
    adamPoint = zip(timeDiff_adam, acc_adam)
    torchSGDPoint = zip(timeDiff_torchSGD, acc_sgd)
    theanoSGDPoint = zip(timeDiff_theanoSGD, acc_theano)

    # plt.title('Accuracy vs. time')
    plt.plot(timeDiff_adam, acc_adam, color='green', label='PyTorch_ADAM')
    plt.plot(timeDiff_torchSGD, acc_sgd, color='red', label='PyTorch_SGD')
    plt.plot(timeDiff_theanoSGD, acc_theano,  color='skyblue', label='Theano_SGD')
    plt.legend()
    plt.xlabel('Time in Minute')
    plt.ylabel('Accuracy Rate')
    plt.savefig('result.png')

if __name__ == "__main__":
    # sns.set()
    timeLine_adam, acc_adam, epoch_adam = loadResult(resultPath_torch154DAM)
    timeLine_sgd, acc_sgd, epoch_sgd = loadResult(resultPath_torch152SGD)
    timeLine_theano, acc_theano, epoch_theano = loadResult(resultPath_theano152SGD)
    plotResult(timeLine_adam, acc_adam, epoch_adam, timeLine_sgd, acc_sgd, epoch_sgd, timeLine_theano, acc_theano, epoch_theano)