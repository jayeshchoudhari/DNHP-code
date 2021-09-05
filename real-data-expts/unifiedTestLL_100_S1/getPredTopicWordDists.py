# Compute Topic Word Distributions from Mode Topic Assignment

from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter

vocabSize = 33000
numTopics = 25
invalidTopicId = 1000

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

print len(topicAssignments)

docsFile = open("../../centralFiles/dataFile.txt")

linenum = 0

for line in docsFile:
	flds = line.strip().split()

	# print "Line num: ",  linenum, " ", line.strip()
	assignedTopic = topicAssignments[linenum]

	if assignedTopic < invalidTopicId:

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
	normalization += vocabSize
	topicDist = [(x+1)/normalization for x in eachTopic]
	topicWordDists.append(topicDist)

	# print "Distbtn -- ", topicDist

	writeDist = str(topicId) + " " + " ".join([str(x) for x in topicDist]) + "\n"

	predTopicDistFile.write(writeDist)

	topicId += 1

predTopicDistFile.close()
