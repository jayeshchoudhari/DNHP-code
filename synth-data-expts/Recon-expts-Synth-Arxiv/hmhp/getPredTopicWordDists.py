# Compute Topic Word Distributions from Mode Topic Assignment

from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter

vocabSize = 500
numTopics = 10

topicWordCounts = []

for i in range(numTopics):
	temp = [0]*vocabSize
	topicWordCounts.append(temp)

modeTAFile = open("modeTopicAssignment.txt")
topicAssignments = []

for line in modeTAFile:

	flds = line.strip().split()
	# eid = int(flds[0].strip())
	tid = int(flds[1].strip())
	topicAssignments.append(tid)

modeTAFile.close()


docsFile = open("../centralFiles/docs_1L.txt")

linenum = 0

for line in docsFile:
	flds = line.strip().split()

	assignedTopic = topicAssignments[linenum]

	for strWord in flds:
		wordId = int(strWord.strip())
		topicWordCounts[assignedTopic][wordId] += 1

	linenum += 1

# print topicWordCounts

topicWordDists = []

predTopicDistFile = open("predTopicDistFile.txt", "w")

topicId = 0

for eachTopic in topicWordCounts:

	# print "eachtopic --", eachTopic
	normalization = sum(eachTopic)
	topicDist = [x/normalization for x in eachTopic]
	topicWordDists.append(topicDist)

	# print "Distbtn -- ", topicDist

	writeDist = str(topicId) + " " + " ".join([str(x) for x in topicDist]) + "\n"

	predTopicDistFile.write(writeDist)

	topicId += 1

predTopicDistFile.close()
