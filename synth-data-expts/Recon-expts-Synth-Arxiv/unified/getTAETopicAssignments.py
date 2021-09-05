# Check for TAE of the topic assignments...

from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter


def readTopicWordDist(fileName):

	localTopicDist = []

	fptr = open(fileName)
	
	for line in fptr:

		flds = line.strip().split()
		tempDist = [float(x.strip()) for x in flds[1:]]

		localTopicDist.append(tempDist)

	return localTopicDist


predTopicDist = readTopicWordDist("predTopicDistFile.txt")
origTopicDist = readTopicWordDist("../centralFiles/topicWordDistribution_dirichlet.txt")

# print "Got Topic Dists", predTopicDist, "orig Dist --", origTopicDist

modeTAFile = open("modeTopicAssignment.txt")
predTopicAssignments = []

for line in modeTAFile:
	flds = line.strip().split()
	tid = int(flds[1].strip())
	predTopicAssignments.append(tid)

modeTAFile.close()

print "Got pred topic assignments"

origEventsFile = open("../centralFiles/events_1L.txt")
origTopicAssignments = []

for line in origEventsFile:
	flds = line.strip().split()
	origTopicAssignments.append(int(flds[3].strip()))

origEventsFile.close()

print "Got orig topic assignments"

totalL1Diff = 0
l1DiffList = []

for i in range(len(predTopicAssignments)):

	predTopic = predTopicAssignments[i]
	origTopic = origTopicAssignments[i]

	# print predTopic, origTopic

	pTDist = predTopicDist[predTopic]
	oTDist = origTopicDist[origTopic]

	# print pTDist, oTDist

	l1DiffVal = sum([abs(pTDist[j] - oTDist[j]) for j in range(len(pTDist))])
	# l1DiffVal = sum([(pTDist[j] - oTDist[j])**2 for j in range(len(pTDist))])

	totalL1Diff += l1DiffVal

	l1DiffList.append(l1DiffVal)


print "Total L1 Diff = ", totalL1Diff
print "Avg L1 Diff = ", totalL1Diff / len(predTopicAssignments)
print "Max L1 = ", max(l1DiffList)
print "Median L1 = ", np.median(l1DiffList)
print "Min L1 = ", min(l1DiffList) 
print "Var L1 = ", np.var(l1DiffList) 
print "Std Dev. = ", np.std(l1DiffList)

# print "Max L1 Index = ", l1DiffList.index(max(l1DiffList))
# maxL1Index = l1DiffList.index(max(l1DiffList))
# print predTopicAssignments[maxL1Index], origTopicAssignments[maxL1Index]
# print predTopicDist[predTopicAssignments[maxL1Index]], origTopicDist[origTopicAssignments[maxL1Index]]
