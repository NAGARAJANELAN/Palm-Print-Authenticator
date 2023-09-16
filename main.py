from palmPrintCollector import palmPrintCollector
from palmMatchScorer import palmMatchScore
from palmPrintAuthenticator import palmPrintAuthenticate

DATA_DIR = "./Palmprint/training/"


# cd to current folder before running
def main():
    # palmPrintCollector('left')
    palmPrintAuthenticate("left")


main()
