from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter


eventsFile = open("events_1L.txt", "r")
allParents = []

for line in eventsFile:

	flds = line.strip().split()
	eTime = flds[0]
	eNode = int(flds[1])
	eParent = int(flds[2])
	eTopic = int(flds[3])
	eLevel = int(flds[4])

	allParents.append(eParent)


eventsFile.close()



top100ParFile = open("top100Parents.txt", "r")

linenum = 0
presentCount = 0
totalEventsWithPar = 0

for line in top100ParFile:
	
	flds = line.strip().split()
	count = int(flds[0].strip())
	eid = int(flds[1].strip())

	if count > 0:

		totalEventsWithPar += 1

		flag = 0
		origPar = allParents[linenum]

		for candParStr in flds[2:]:
			candPar = int(candParStr.strip())

			if candPar == origPar:
				flag = 1
				break

		if flag == 1:
			presentCount += 1


	linenum	+= 1


print "presentCount = ", presentCount
print "total par events = ", totalEventsWithPar