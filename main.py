from palmPrintCollector import palmPrintCollector
from palmMatchScorer import palmMatchScore
from palmPrintAuthenticator import palmPrintAuthenticate

DATA_DIR = "./Palmprint/training/"


# cd to current folder before running
def main():
    palmPrintAuthenticate("left")
    
if __name__ == "__main__":
    main()
