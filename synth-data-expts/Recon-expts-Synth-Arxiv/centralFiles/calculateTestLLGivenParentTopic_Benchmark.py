from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from decimal import Decimal

maxTrainLevel = 2
numTopics = 10

def getUserUserInfluence(fileName):

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
			j = j + 2

	wuvfile.close()
	return tempWuv


def getAllDocs(fileName):

	fptr = open(fileName)
	localAllDocs = []

	for line in fptr:
		thisDoc = []
		flds = line.strip().split()
		for eachWord in flds:
			thisDoc.append(int(eachWord.strip()))

		localAllDocs.append(thisDoc)

	return localAllDocs

def getTopicWordDistributions(fileName):

	fptr = open(fileName)
	
	localTopicWordDist = defaultdict(list)

	for line in fptr:

		flds = line.strip().split()
		topicId = int(flds[0].strip())

		for i in flds[1:]:
			probVal = float(i.strip())
			localTopicWordDist[topicId].append(probVal)

	return localTopicWordDist


def getContentProb(wordsSet, topicId):

	localTotalWordProb = 1

	for eachWord in wordsSet:
		localTotalWordProb *= topicWordDist[topicId][eachWord]

	return localTotalWordProb

print("Getting user user influence map...")
W = getUserUserInfluence("userUserInfluence.txt")
print("Got user-user influence matrix", len(W))

# nodes = list(W.keys())
# numNodes = len(nodes)

print("Getting Topic-Topic influence map...")
Tau = getUserUserInfluence("topicTopicInfluenceFile.txt")
print("Got Topic-Topic influence matrix", len(Tau))


print("Getting Topic Word Distributions...")
topicWordDist = getTopicWordDistributions("topicWordDistribution_dirichlet.txt")
print("Got Topic Word Distributions...")

allDocs = getAllDocs("docs_1L.txt")
print("Got all the docs...")


totalLL = 0

allEvents = []

eventsfptr = open("events_1L.txt", "r")

llFilePtr = open("./benchmarkTestLL.txt", "w")
numLLEvents = 0


linenum = 0
for eventLine in eventsfptr:

	# eventLine = eventLine.strip()
	
	eventsFlds = eventLine.strip().split()

	lInfo = int(eventsFlds[4].strip())
		
	eTime = Decimal(eventsFlds[0].strip())
	eUid = int(eventsFlds[1].strip())
	ePid = int(eventsFlds[2].strip())
	eTopic = int(eventsFlds[3].strip())
	eLevel = int(eventsFlds[4].strip())

	allEvents.append([eTime, eUid, ePid, eTopic, eLevel])

	currDoc = allDocs[linenum]

	if lInfo > maxTrainLevel and len(currDoc) > 0:
		parTime = allEvents[ePid][0]
		parNode = allEvents[ePid][1]
		parTopic = allEvents[ePid][3]

		contentProb = getContentProb(currDoc, eTopic)

		delT = float(parTime - eTime)

		WuvTaukk = W[parNode][eUid] * Tau[parTopic][eTopic]

		eventLL = np.log(WuvTaukk) - WuvTaukk + float(delT) + np.log(contentProb)
		totalLL += eventLL

		llFilePtr.write(str(linenum) + " " + str(eventLL) + " " + str(eventLL) + "\n")

		numLLEvents += 1

	linenum += 1

print "total LL = ", totalLL, "num LL Events = ", numLLEvents
print "Avg LL = ", totalLL / numLLEvents
llFilePtr.close()
