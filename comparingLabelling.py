import csv
import numpy as np
#trueFileName = "045-manualLabelling.txt"
#createdFileName = "045-manualLabelling - distance to reprojected bolts.txt"
methodName = "matching to reprojected features"

def compare(inputFileType, createdFileName, trueFileName):
    print("\nRunning matching script to determine label accuracy.\n")
    if(inputFileType == "manualLabelling"):
        createdFile = open(createdFileName, "r")


        trueFeatures = []
        with open(trueFileName, mode='r') as trueFile:
            reader = csv.reader(trueFile, delimiter="\t")
            for r in reader:
                trueFeatures.append((r[1],r[2],r[3]))

        labelledFeatures = []
        with open(createdFileName, mode='r') as createdFile:
            reader = csv.reader(createdFile, delimiter="\t")
            for r in reader:
                labelledFeatures.append((r[1],r[2],r[3]))


    if(inputFileType == "imageProcessing"):

        trueFeatures = []
        with open(trueFileName, mode='r') as trueFile:
            reader = csv.reader(trueFile, delimiter="\t")
            for r in reader:
                r[1] = int(float(r[1]))
                r[2] = int(float(r[2]))
                #print(r[1])
                trueFeatures.append((r[0],r[1],r[2]))
        

        labelledFeatures = []
        with open(createdFileName, mode='r') as createdFile:
            reader = csv.reader(createdFile, delimiter="\t")
            for r in reader:
                r[1] = r[1][-2:]
                r[2] = int(r[2])
                r[3] = int(r[3])
                labelledFeatures.append((r[1],r[2],r[3]))        

    matches = 0
    correctFeatures = []
    for labelledFeature in labelledFeatures:
        for trueFeature in trueFeatures:
            if(labelledFeature == trueFeature):
                matches = matches+1
                print(labelledFeature,trueFeature)
                correctFeatures.append(trueFeature)
            #feature_locations = {r[0]: np.array([r[1], r[2], r[3]]) for r in reader}
        #print(feature_locations)
    print("UNMATCHED:")
        #print(featureLabelArray[2])
    incorrectFeatures = []
    for trueFeature in trueFeatures:
        if trueFeature not in correctFeatures:
            print(trueFeature)
            incorrectFeatures.append(trueFeature)

    

    print(methodName+":", "Correctly found", matches, "features out of", len(trueFeatures), ",",(matches / len(labelledFeatures)) *100,"% correct.")
    return incorrectFeatures
