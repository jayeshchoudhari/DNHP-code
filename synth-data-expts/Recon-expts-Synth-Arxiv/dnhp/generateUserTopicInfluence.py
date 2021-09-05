# generate synthetic Qovk 

from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

alphaAllEdges = 1.0
betaAllEdges = 0.02

numUsers = 151
numTopics = 25

qvkWriteFile = open("userTopicInfluence.txt", "w")


for vNode in range(numUsers):

	qstr = str(numTopics) + " " + str(vNode) + " "
	qvkList = []
	for kTopic in range(numTopics):

		qvkVal = np.random.gamma(alphaAllEdges, betaAllEdges, 1)[0]

		qvkList.append(kTopic)
		qvkList.append(qvkVal)

	qstr += " ".join([str(x) for x in qvkList]) + "\n"

	qvkWriteFile.write(qstr)


qvkWriteFile.close()