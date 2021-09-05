from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

numTopics = 10

def getParentChildCounts(fileName):

	eventsFile = open(fileName)

	parentIds = []
	allEvents = []
	localParentChildCount = defaultdict(lambda: defaultdict(int))

	for line in eventsFile:

		flds = line.strip().split()
		eTime = flds[0]
		eNode = int(flds[1])
		eParent = int(flds[2])
		eTopic = int(flds[3])
		eLevel = int(flds[4])

		tempEvent = [eTime, eNode, eParent, eTopic]
		allEvents.append(tempEvent)

		parentIds.append(eParent)

		if eParent > -1:

			pNode = allEvents[eParent][1]

			localParentChildCount[pNode][eNode] += 1

	eventsFile.close()


	localMaxCount = 0
	for u in localParentChildCount:
		for v in localParentChildCount[u]:
			if localParentChildCount[u][v] > localMaxCount:
				localMaxCount = localParentChildCount[u][v]

	return localParentChildCount, localMaxCount


def getUserUserInfluence(fileName):

	totalInfluence = 0

	wuvfile = open(fileName)

	tempWuv = {}

	for line in wuvfile:
		splitLine = line.split()
		count = int(splitLine[0].strip())
		uNode = int(splitLine[1].strip())

		tempWuv[uNode] = {}
		# followers[uNode] = []

		j = 2
		while j < len(splitLine):
			vNode = int(splitLine[j].strip())
			uvInf = float(splitLine[j+1].strip())
						
			tempWuv[uNode][vNode] = uvInf
			totalInfluence += uvInf
			j = j + 2


	wuvfile.close()
	return [tempWuv, totalInfluence]


def getEffectiveWuvValue(W, totalKKInf):

	localDict = defaultdict(lambda: defaultdict(float))

	for uNode in W:

		for vNode in W[uNode]:

			eVal = W[uNode][vNode] * totalKKInf

			localDict[uNode][vNode] = eVal

	return localDict


eventsFileName = "../centralFiles/events_1L.txt"
parentChildCount, maxCount = getParentChildCounts(eventsFileName)
print("Got parent child (Nuv) counts... -- maxCount = ", maxCount)

WPred, totalUVInfPred = getUserUserInfluence("userUserInf.txt")
print("Got user-user influence matrix", len(WPred), totalUVInfPred)

effectiveWuvPred = getEffectiveWuvValue(WPred, numTopics)

WOrig, totalUVInfOrig = getUserUserInfluence("../centralFiles/userUserInfluence.txt")
print("Got Orig user-user influence matrix", len(WOrig), numTopics)

TauOrig, totalKKInfOrig = getUserUserInfluence("../centralFiles/topicTopicInfluenceFile.txt")
print("Got Orig topic topic influence matrix", len(TauOrig), totalKKInfOrig)

effectiveWuvOrig = getEffectiveWuvValue(WOrig, totalKKInfOrig)

countThresholds = []
totalMSEErr = []
avgRelative = []
totalRelErr = []
edgeCounts = []

totalMSEErrAll = 0
totalRelErrAll = 0
edgeCountsAll = 0

nuvThresholds = [100, 500, 1000, 2000]
mseAboveX = [0, 0, 0, 0]
relAboveX = [0, 0, 0, 0]
edgeCountsAboveX = [0, 0, 0, 0]


for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 101]:
	countVal = i * 0.01 * maxCount
	countThresholds.append(countVal)
	totalMSEErr.append(0)
	avgRelative.append(0)
	totalRelErr.append(0)
	edgeCounts.append(0)


for uNode in WOrig:

	for vNode in WOrig[uNode]:

		nuvCount = parentChildCount[uNode][vNode]

		origVal = effectiveWuvOrig[uNode][vNode]
		predVal = effectiveWuvPred[uNode][vNode]

        # err = abs(origVal - predVal)**2
		err = abs(origVal - predVal)
		
		relErr = abs(origVal - predVal) / origVal

		for i in range(len(countThresholds)):

			if nuvCount < countThresholds[i]:

				totalMSEErr[i] += err
				totalRelErr[i] += relErr
				edgeCounts[i] += 1

		# print uNode, vNode, origVal, predVal, err

		totalMSEErrAll += err
		totalRelErrAll += relErr

		edgeCountsAll += 1

		for j in range(len(nuvThresholds)):

			if nuvCount > nuvThresholds[j]:
				mseAboveX[j] += err
				relAboveX[j] += relErr
				edgeCountsAboveX[j] += 1


print totalMSEErrAll
print totalMSEErrAll/edgeCountsAll
print totalRelErrAll
print totalRelErrAll/edgeCountsAll
print edgeCountsAll


print "TAE = ", totalMSEErr
print "Avg TAE = ", [totalMSEErr[i]/edgeCounts[i] if edgeCounts[i] > 0 else 0 for i in range(len(totalMSEErr))]
print "Rel = ", totalRelErr
print "Avg Rel = ", [totalRelErr[i]/edgeCounts[i] if edgeCounts[i] > 0 else 0 for i in range(len(totalRelErr))]

print edgeCounts
print countThresholds

print "TAE Above X= ", mseAboveX
print "Avg TAE Above X = ", [mseAboveX[i]/edgeCountsAboveX[i] if edgeCountsAboveX[i] > 0 else 0 for i in range(len(mseAboveX))]
print "Rel Above X= ", relAboveX
print "Avg Rel Above X = ", [relAboveX[i]/edgeCountsAboveX[i] if edgeCountsAboveX[i] > 0 else 0 for i in range(len(relAboveX))]

print edgeCountsAboveX