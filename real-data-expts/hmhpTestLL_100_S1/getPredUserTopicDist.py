# Compute Topic Word Distributions from Mode Topic Assignment

from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter

maxTrainLevel = 3
numTopics = 25
invalidTopicId = 1000
maxNumNodes = 151;
userTopicCounts = []

for i in range(maxNumNodes):
	temp = [0]*numTopics
	userTopicCounts.append(temp)

modeTAFile = open("modeTopicAssignment.txt")
topicAssignments = []

for line in modeTAFile:

	flds = line.strip().split()
	# eid = int(flds[0].strip())
	tid = int(flds[1].strip())
	topicAssignments.append(tid)

modeTAFile.close()

print len(topicAssignments)

eventsFile = open("../../centralFiles/events_scaledTime_EventSelect_0_testThresh_3_set_1.txt")

linenum = 0

for line in eventsFile:
	flds = line.strip().split()

	eUId = int(flds[1].strip())
	eLevel = int(flds[4].strip())

	assignedTopic = topicAssignments[linenum]

	if assignedTopic < invalidTopicId:
		userTopicCounts[eUId][assignedTopic] += 1

	
	if assignedTopic >= numTopics and eLevel <= maxTrainLevel and eLevel >= 0:
		print("There is some issue here....")
		exit(0)

	linenum += 1


# print userTopicCounts

userTopicDists = []

predUserTopicDistFile = open("predUserTopicDistFile.txt", "w")

topicId = 0

for eachUser in userTopicCounts:

	# print "eachtopic --", eachTopic
	normalization = sum(eachUser)
	normalization += numTopics
	topicDist = [(x+1)/normalization for x in eachUser]
	userTopicDists.append(topicDist)

	# print "Distbtn -- ", topicDist

	writeDist = str(topicId) + " " + " ".join([str(x) for x in topicDist]) + "\n"

	predUserTopicDistFile.write(writeDist)

	topicId += 1

predUserTopicDistFile.close()
