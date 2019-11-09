import datetime
import matplotlib.pyplot as plt

resultPath_torch164DAM = '../result/torch16-4-0.001ADAM.txt'
resultPath_torch164SGD = '../result/torch16-4-0.001SGD.txt'
resultPath_theano164SGD = '../result/theano16-4-0.001SGD.txt'
resultPath_torch154DAM = '../result/torch15-4-0.001ADAM.txt'
#resultPath_torch152SGD = '../result/torch15-2-0.005SGD.txt'
#resultPath_theano152SGD = '../result/theano15-2-0.005SGD.txt'

def loadResult(fileName):
    timeLine = []
    acc = []
    with open(fileName, "r") as rFile :
        lines = rFile.readlines()
    for i,line in enumerate(lines):
        if i%2 == 0:
            timeLine.append(lines[i].split(": ")[0].strip())
        else:
            acc.append(float(lines[i].split(",")[1].strip()))
    return timeLine, acc

def plot3(timeLine_adam, acc_adam, timeLine_sgd, acc_sgd, timeLine_theano, acc_theano):
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
            d2 = datetime.datetime.strptime(j, '%Y-%m-%d %H:%M:%S')
            timeDiff[idx].append(divmod((d2-timeStart[idx]).total_seconds(),60*60)[0])
    
    print(timeDiff_adam[-1], timeDiff_torchSGD[-1], timeDiff_theanoSGD[-1])

    plt.plot(timeDiff_adam, acc_adam, color='green', label='PyTorch_ADAM')
    plt.plot(timeDiff_torchSGD, acc_sgd, color='red', label='PyTorch_SGD')
    plt.plot(timeDiff_theanoSGD, acc_theano,  color='skyblue', label='Theano_SGD')
    plt.legend()
    plt.xlabel('Time in Hour')
    plt.ylabel('Accuracy Rate')
    plt.savefig('resultFig3.png')
    
def plot2(timeLine_adam15, acc_adam15, timeLine_adam16, acc_adam16):
    timeDiff_adam15 = []
    timeDiff_adam16 = []
    adam15D1 = datetime.datetime.strptime(timeLine_adam15[0], '%Y-%m-%d %H:%M:%S')
    adam16D1 = datetime.datetime.strptime(timeLine_adam16[0], '%Y-%m-%d %H:%M:%S')
    timeDiff = [timeDiff_adam15, timeDiff_adam16]
    timeLine = [timeLine_adam15, timeLine_adam16]
    timeStart = [adam15D1, adam16D1]
    for idx,t in enumerate(timeLine):
        for j in t:
            d2 = datetime.datetime.strptime(j, '%Y-%m-%d %H:%M:%S')
            timeDiff[idx].append(divmod((d2-timeStart[idx]).total_seconds(),60)[0])
    print(timeDiff_adam15[-1], timeDiff_adam16[-1])
    plt.plot(timeDiff_adam15, acc_adam15, color='green', label='Twitter15')
    plt.plot(timeDiff_adam16, acc_adam16, color='red', label='Twitter16')
    plt.legend()
    plt.xlabel('Time in Minute')
    plt.ylabel('Accuracy Rate')
    plt.savefig('resultFig2.png')

if __name__ == "__main__":
    timeLine_adam15, acc_adam15 = loadResult(resultPath_torch154DAM)
    timeLine_adam16, acc_adam16 = loadResult(resultPath_torch164DAM)
#    timeLine_sgd15, acc_sgd15 = loadResult(resultPath_torch152SGD)
    timeLine_sgd16, acc_sgd16 = loadResult(resultPath_torch164SGD)
    timeLine_theano, acc_theano = loadResult(resultPath_theano164SGD)
    plot3(timeLine_adam16, acc_adam16, timeLine_sgd16, acc_sgd16, timeLine_theano, acc_theano)
#    plot2(timeLine_adam15, acc_adam15, timeLine_adam16, acc_adam16)