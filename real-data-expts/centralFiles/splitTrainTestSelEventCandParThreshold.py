# interleave level...

from __future__ import division
import sys
import numpy as np
import random
import math
from collections import defaultdict
from collections import Counter

setNumber = int(sys.argv[1])
eventSelectionProb = 0.0
# probOfLastLevelEvent = 1.0/5
# probOfLastBut1LevelEvent = 1.0/2
lastLevelValue = 4
threshold = 0.5
lB1LevelThreshold = 0.5


# A function to generate a random permutation of arr[] 
def randomize (arr, n): 
    # Start from the last element and swap one by one. We don't 
    # need to run for the first element that's why i > 0 
    for i in range(n-1,0,-1):
        # Pick a random index from 0 to i 
        j = random.randint(0, i+1)
  
        # Swap arr[i] with the element at random index 
        arr[i],arr[j] = arr[j],arr[i] 
    return arr 


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


print("Get All candidate Parents...")
top100Parents = getAllCandidateParents("top100Parents.txt")
print("Got all candidate parents...", len(top100Parents))


fptr = open("eventsFile_scaledTime.txt")
localAllEvents = []
eventClass = defaultdict(int)

testCount = 0
lastBut1Level = 0
linenum = 0

selEvents = []

for line in fptr:

	eventInTestFlag = 1

	eventsFlds = line.strip().split()

	eTime = eventsFlds[0].strip()
	eUid = eventsFlds[1].strip()
	ePid = eventsFlds[2].strip()
	eTopic  = eventsFlds[3].strip()

	eSelProb = np.random.sample()

	if eSelProb < eventSelectionProb:
		eLevel = -1
	else:
		eLevel = 1
		selEvents.append(linenum)

	localAllEvents.append([eTime, eUid, ePid, eTopic, eLevel])	

	linenum += 1  

fptr.close()


selEventsLen = len(selEvents)
arr = range(selEventsLen) 
n = len(arr) 
shuffledArr = randomize(arr, n)

print "Num of selected events = ", selEventsLen

for eachIndex in shuffledArr:

	selEventIndex = selEvents[eachIndex]

	eventTuple = localAllEvents[selEventIndex]  

	candParentSet = top100Parents[selEventIndex]
	totalCandPar = len(candParentSet)

	if totalCandPar == 0:
		eventInTestFlag = 0
		eLevel = str(lastLevelValue - 2)
	else:
		countInTest = 0
		presentParCount = 0
		for eachPar in candParentSet:
			# count the cand parents in test...
			if eventClass.get(eachPar, -1) != -1:
				presentParCount += 1
				if eventClass[eachPar] == 1:
					countInTest += 1

		if countInTest > 0:
			fracInTest = (countInTest * 1.0) / presentParCount
		else:
			fracInTest = 0.0 

		if fracInTest > threshold:
			# put the event in train...
			eventInTestFlag = 0
			prob = np.random.sample()
			if prob < lB1LevelThreshold:
				eLevel = str(lastLevelValue - 1)
				lastBut1Level += 1
			else:
				eLevel = str(lastLevelValue - 2)
		else:
			# put the event in test...
			eventInTestFlag = 1
			eLevel = str(lastLevelValue)
			testCount += 1
	

	localAllEvents[selEventIndex][4] = eLevel
	# localAllEvents.append([eTime, eUid, ePid, eTopic, eLevel])

	# eventClass.append(eventInTestFlag)
	eventClass[selEventIndex] = eventInTestFlag

	# linenum += 1


print "Test Count = ", testCount
print "last But 1 Level = ", lastBut1Level

# levelEventsFile = open("events_scaledTime_interLeaveLastLevel_Set_1.txt", "w")
levelEventsFile = open("events_scaledTime_EventSelect_" + str(int(eventSelectionProb*10)) + "_testThresh_" + str(int(threshold*10)) + "_set_" + str(setNumber) + ".txt", "w")

for eachEvent in localAllEvents:

	writeStr = " ".join([str(x) for x in eachEvent]) + "\n"

	levelEventsFile.write(writeStr)

levelEventsFile.close()
