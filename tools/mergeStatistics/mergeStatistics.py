import json
import os
from sys import argv


# gets the right separator based on the os
def getRightSeparator():
    if len(os.path.abspath(__file__).split('/')) > 1:
        separator = '/'
    else:
        separator = '\\'
    return separator


# initialize the key
def initializeKey(finalSpatial, key):
    finalSpatial[key] = {}
    return finalSpatial


def getAbsolutePath(relative_path):
    return os.path.abspath(os.path.join(os.getcwd(), relative_path))


# path to the files and creation of the original dictionaries
separator = getRightSeparator()
with open(getAbsolutePath(argv[1]), 'r') as f:
    inputMap = json.load(f)
# chose the name of the folders to compare
mapList = []
# chose a probability for each folder
weightsList = []
checkWeight = 0.0
for i in range(0, len(inputMap)):
    with open(getAbsolutePath(inputMap[i]["file"]), 'r') as f:
        mapList.append(json.load(f))
    weightsList.append(inputMap[i]["weight"])
    checkWeight+=weightsList[i]

if checkWeight !=  1:
    print("Error! sum of the weights must be one!")
    exit(-1)
# comparing the dictionaries
alreadySeenKeys = {}

finalSpatial = {}
# entire dictionary
for i_map in range(0, len(mapList) - 1):
    statsMap = mapList[i_map]
    i_weight = float(weightsList[i_map])
    # number of errors "0", "2"...
    for nErrorKey, nErrorMap in statsMap.items():

        # initialize finalSpatial
        if nErrorKey not in finalSpatial:
            finalSpatial = initializeKey(finalSpatial, nErrorKey)
            finalSpatial[nErrorKey] = initializeKey(finalSpatial[nErrorKey], "FF")
            finalSpatial[nErrorKey] = initializeKey(finalSpatial[nErrorKey], "PF")
        # initialize already seen keys
        if nErrorKey not in alreadySeenKeys:
            alreadySeenKeys[nErrorKey] = {}
            alreadySeenKeys[nErrorKey]["FF"] = []
            alreadySeenKeys[nErrorKey]["PF"] = []
        # FF map iterations
        for errorPatternKey, probability in nErrorMap["FF"].items():
            if errorPatternKey not in alreadySeenKeys[nErrorKey]["FF"]:
                nMapSharingKey = i_weight
                probSum = float(i_weight) * float(probability)

                for k in range(i_map + 1, len(mapList)):
                    if nErrorKey in mapList[k].keys():
                        if errorPatternKey in mapList[k][nErrorKey]["FF"]:
                            nMapSharingKey += float(weightsList[k])
                            probSum += float(weightsList[k]) * float(mapList[k][nErrorKey]["FF"][errorPatternKey])

                if errorPatternKey not in finalSpatial[nErrorKey]["FF"]:
                    finalSpatial[nErrorKey]["FF"] = initializeKey(finalSpatial[nErrorKey]["FF"], errorPatternKey)
                finalSpatial[nErrorKey]["FF"][errorPatternKey] = float(probSum) / float(nMapSharingKey)
            alreadySeenKeys[nErrorKey]["FF"].append(errorPatternKey)

        # PF map iteration
        for key, probability in nErrorMap["PF"].items():
            if key not in alreadySeenKeys[key]["PF"]:
                if key != "MAX":
                    nMapSharingKey = i_weight
                    probSum = i_weight * probability
                    for k in range(i_map + 1, len(mapList)):
                        if key in mapList[k][nErrorKey]:
                            nMapSharingKey += float(weightsList[k])
                            probSum += float(weightsList[k]) * float(mapList[k][nErrorKey][key])

                    if key not in finalSpatial[nErrorKey]["FF"]:
                        finalSpatial[nErrorKey]["PF"] = initializeKey(finalSpatial[nErrorKey]["PF"], key)

                    finalSpatial[nErrorKey]["PF"][key] = float(probSum) / float(nMapSharingKey)
                else:
                    # in this case probability values describe the MAX value and not a probability
                    finalSpatial[nErrorKey]["PF"]["MAX"] = probability
                    for k in range(i_map + 1, len(mapList)):
                        if "MAX" in mapList[k][nErrorKey]["PF"]:
                            if finalSpatial[nErrorKey]["PF"]["MAX"] < mapList[k][nErrorKey]["PF"]["MAX"]:
                                finalSpatial[nErrorKey]["PF"]["MAX"] = mapList[k][nErrorKey]["PF"]["MAX"]

                alreadySeenKeys[key]["PF"].append(key)

# Save results
file = open(getAbsolutePath( "results" + separator + "merged_spatial.json"), "w")
file.write(json.dumps(finalSpatial))
file.close()

print("merge finished correctly!")
