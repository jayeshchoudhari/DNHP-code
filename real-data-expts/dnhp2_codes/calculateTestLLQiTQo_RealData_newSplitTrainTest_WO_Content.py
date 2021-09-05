from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from decimal import Decimal

maxTrainLevel = 3
numTopics = 100
dataFileName = "../../centralFiles/events_scaledTime_EventSelect_0_testThresh_5_set_1.txt"

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


def getAssignedParentsTopicsFromTrain(fileName):

	fptr = open(fileName)
	localAllAssignment = []

	for line in fptr:

		flds = line.strip().split()
		eId = int(flds[0].strip())
		assignedId = int(flds[1].strip())
		localAllAssignment.append(assignedId)

	fptr.close()

	return localAllAssignment


def getAllEvents(fileName):

	fptr = open(fileName)
	localAllEvents = []

	for line in fptr:

		eventsFlds = line.strip().split()

		eTime = Decimal(eventsFlds[0].strip())
		eUid = int(eventsFlds[1].strip())
		ePid = int(eventsFlds[2].strip())
		eTopic  = int(eventsFlds[3].strip())
		eLevel = int(eventsFlds[4].strip())

		localAllEvents.append([eTime, eUid, ePid, eTopic, eLevel])

	return localAllEvents


def getAllCandidateParents(fileName):

	fptr = open(fileName)
	localTop100Parents = []

	for line in fptr:

		flds = line.strip().split()
		count = int(flds[0].strip())
		eId = int(flds[1].strip())

		parList = [int(x.strip()) for x in flds[2:]]

		localTop100Parents.append(parList)

	return localTop100Parents


def getUserTopicBaseRates(fileName):

	fptr = open(fileName)
	localBaseRates = defaultdict(float)

	for line in fptr:

		flds = line.strip().split()
		uId = int(flds[0].strip())
		bRate = float(flds[1].strip())
		localBaseRates[uId] = bRate

	fptr.close()

	return localBaseRates


def getContentProb(wordsSet, topicId):

	localTotalWordProb = 1

	for eachWord in wordsSet:
		localTotalWordProb *= topicWordDist[topicId][eachWord]

	return localTotalWordProb


# print("Getting user user influence map...")
# W = getUserUserInfluence("userUserInfAvg.txt")
# print("Got user-user influence matrix", len(W))

print("Getting User-Topic influence map...")
Qin = getUserUserInfluence("userTopicInfInAvg.txt")
print("Got User-Topic influence matrix", len(Qin))

print("Getting Topic-Topic influence map...")
Tau = getUserUserInfluence("topicTopicInfAvg.txt")
print("Got Topic-Topic influence matrix", len(Tau))

print("Getting User-Topic influence map...")
Qout = getUserUserInfluence("userTopicInfOutAvg.txt")
print("Got User-Topic influence matrix", len(Qout))

# print("Getting Topic Word Distributions...")
# topicWordDist = getTopicWordDistributions("predTopicDistFile.txt")
# print("Got Topic Word Distributions...")

print("Getting all docs in memory...")
allDocs = getAllDocs("../../centralFiles/dataFile.txt")
print("Got all the docs...")

print("Get Assigned Topics...")
allEventsTopic = getAssignedParentsTopicsFromTrain("modeTopicAssignment.txt")
print("Got Assigned Topics...")

print("Get Assigned Parents...")
allEventsParent = getAssignedParentsTopicsFromTrain("modeParentAssignment.txt")
print("Got Assigned Parents...")

print("Get User base rate...")
userBaseRates = getUserTopicBaseRates("userBaseRate.txt")
print("Got User base rate...", userBaseRates)

print("Get topic base rate...")
topicBaseRates = getUserTopicBaseRates("topicBaseRate.txt")
print("Got topic base rate...", topicBaseRates)

print("Get all events...")
allEvents = getAllEvents(dataFileName)
print("Got all the events...", len(allEvents))

print("Get All candidate Parents...")
top100Parents = getAllCandidateParents("../../centralFiles/top100Parents.txt")
print("Got all candidate parents...", len(top100Parents))

totalTime = allEvents[len(allEvents)-1][0] - allEvents[0][0]
print("Total Time = ", totalTime)

totalLL = 0

llFilePtr = open("./llFileWO.txt", "w")
numLLEvents = 0

emptyDocCount = 0
trainCount = 0
linenum = 0

for eachEid in range(len(allEvents)):

	# print("I am at least coming in here....\n")

	lInfo = allEvents[eachEid][4]
		
	eTime = allEvents[eachEid][0]
	eUid = allEvents[eachEid][1]
	
	currDoc = allDocs[eachEid]

	# assigned parent and topic
	if lInfo > maxTrainLevel and len(currDoc) > 0:

		totalProb = 0

		for topicId in range(numTopics):

			eTopic  = topicId
			# contentProb = getContentProb(currDoc, eTopic)
			contentProb = 1.0
			
			eventCandParents = top100Parents[eachEid]

			totalParProb = 0

			for eachParId in eventCandParents:

				if allEvents[eachParId][4] >= 0 and allEvents[eachParId][4] <= maxTrainLevel:

					parTime = allEvents[eachParId][0]
					parNode = allEvents[eachParId][1]
					# assgined topic of parent...
					parTopic = allEventsTopic[eachParId]
					
					delT = float(parTime - eTime)

					# WuvTaukk = W[parNode][eUid] * Tau[parTopic][topicId]
					# WuvTaukk = W[parNode][eUid] * Tau[parTopic][topicId] * Qout[eUid][topicId]
					QiTQo = Qin[eUid][parTopic] * Tau[parTopic][topicId] * Qout[eUid][topicId]

					parProb = np.exp(-QiTQo) * QiTQo * np.exp(float(delT))

					totalParProb += parProb

			# selfParent Prob...
			uBRate = userBaseRates[eUid]
			tBRate = topicBaseRates[eTopic]

			muUMuk = Decimal(uBRate * tBRate)
			muUMukTotalTime = muUMuk * totalTime

			selfParProb = np.exp(-muUMukTotalTime) * muUMuk

			totalParProb += float(selfParProb)

			eachTopicAllParProb = contentProb * totalParProb
 
			totalProb += eachTopicAllParProb

		eventLL = np.log(totalProb)
		totalLL += eventLL

		llFilePtr.write(str(eachEid) + " " + str(eventLL) + " " + str(eventLL) + "\n")

		numLLEvents += 1


	elif lInfo > maxTrainLevel and len(currDoc) == 0:
		emptyDocCount += 1

	elif lInfo >= 0 and lInfo <= maxTrainLevel:
		trainCount += 1

	linenum += 1

	if linenum % 10000 == 0:
		print linenum

print "Empty docs in test set = ", emptyDocCount
print "Total train = ", trainCount 
print "total LL = ", totalLL, "num LL Events = ", numLLEvents
print "Avg LL = ", totalLL / numLLEvents
llFilePtr.close()