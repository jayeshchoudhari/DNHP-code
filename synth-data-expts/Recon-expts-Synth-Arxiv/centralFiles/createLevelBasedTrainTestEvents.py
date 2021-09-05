from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

maxTrainLevel = 2

eventsFile = "events_1L.txt"
docsFile = "docs_1L.txt"

trainEventsFilePtr = open("events_train_2_Levels.txt", "w")
trainDocsFilePtr = open("docs_train_2_Levels.txt", "w")

testEventsFilePtr = open("events_test_2_Levels.txt", "w")
testDocsFilePtr = open("docs_test_2_Levels.txt", "w")

newEid = defaultdict(int)

with open(eventsFile, 'r') as eventsfptr, open(docsFile, 'r') as docsfptr:

	linenum = 0
	testCount = 0
	# eventLine = eventsfptr.next()
	for eventLine in eventsfptr:
		docLine = docsfptr.next()

		eventLine = eventLine.strip()
		docLine = docLine.strip()

		eventsFlds = eventLine.strip().split()
		lInfo = int(eventsFlds[4].strip())

		if lInfo <= maxTrainLevel:

			eTime = eventsFlds[0].strip()
			eUid = int(eventsFlds[1].strip())
			ePid = int(eventsFlds[2].strip())
			eTopic = int(eventsFlds[3].strip())
			eLevel = int(eventsFlds[4].strip())

			newEid[linenum] = linenum - testCount

			if ePid > -1:
				ePid = newEid[ePid]
				writeStr = eTime + " " + str(eUid) + " " + str(ePid) + " " + str(eTopic) + " " + str(eLevel)
				trainEventsFilePtr.write(writeStr + "\n")

			else:
				trainEventsFilePtr.write(eventLine + "\n")

			trainDocsFilePtr.write(docLine + "\n")

		else:
			testEventsFilePtr.write(eventLine + "\n")
			testDocsFilePtr.write(docLine + "\n")

			testCount += 1


		linenum += 1

trainEventsFilePtr.close()
trainDocsFilePtr.close()

testEventsFilePtr.close()
testDocsFilePtr.close()
