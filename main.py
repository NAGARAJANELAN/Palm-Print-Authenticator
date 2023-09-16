# from palmPrintCollector import palmPrintCollector
from palmMatchScorer import palmMatchScore
from palmPrintAuthenticator import palmPrintAuthenticate

DATA_DIR = "./Palmprint/training/"


# cd to current folder before running
def main():
    # palmPrintCollector('b')
    # palmPrintCollector('b')
    # print (palmMatchScore('./Palmprint/training/a/pp6.jpg','./Palmprint/training/b/pp7.jpg',ratio=0.80))
    palmPrintAuthenticate("b")


main()
