import sys
import seaborn as sns

#X: time, Y:acc
#X: time, Y:epoch

def _readReport(fileName):

    with open(fileName, "r") as rFile :
        lines = rFile.readlines()

    userID = lines[0].split(":")[1].strip()

    columnHeader = lines[2].split()[1:]
    elements = []
    for line in lines[4:]:
        elements.append(line.split())

    rowHeader = [e[0] for e in elements]
    body = [e[1:] for e in elements]

    return userID, columnHeader, rowHeader, body


if __name__ == "__main__":
    fileName = r'D:\Temp\ECE364\Prelab04\reports\report_0B8AE659-99BE-45F5-9DCE-FA807D15DC18.dat'

    userID, columnHeader, rowHeader, body = _readReport(fileName)
