from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter

adjList = defaultdict(lambda:defaultdict(int))

#read graph Edges
ttGraphFile = open("../centralFiles/topicTopicGraph.txt", "r")

edgeList = []
allNodes = defaultdict(int)

for line in ttGraphFile:
	flds = line.strip().split()
	count = int(flds[0].strip())
	sNode = int(flds[1].strip())

	allNodes[sNode] = 1

	for dNode in flds[2:]:
		dNode = int(dNode.strip())
		# edge = (sNode, dNode)
		# edgeList.append(edge)
		adjList[sNode][dNode] = 1


ttGraphFile.close()

topicTopicGammaPriorParam = defaultdict(lambda: defaultdict(float))

for sNode in sorted(adjList.keys()):

	for dNode in sorted(adjList.keys()):

		if adjList[sNode].get(dNode, -1) != -1:
			kParam = 2
			thetaParam = 0.2
			# topicTopicGammaPriorParam[sNode][dNode] = np.random.gamma(kParam, thetaParam, 1)[0]
			topicTopicGammaPriorParam[sNode][dNode] = thetaParam

		else:
			kParam = 1
			thetaParam = 0.001
			# topicTopicGammaPriorParam[sNode][dNode] = np.random.gamma(kParam, thetaParam, 1)[0]
			topicTopicGammaPriorParam[sNode][dNode] = thetaParam


ttPriorFile = open("priorTopicTopicInfluenceFile.txt", "w")

for sNode in sorted(topicTopicGammaPriorParam.keys()):
	writeStr = str(len(topicTopicGammaPriorParam[sNode])) + " " + str(sNode)

	for dNode in sorted(topicTopicGammaPriorParam[sNode].keys()):
		# writeStr += " " + str(dNode) + " " + str(topicTopicGammaPriorParam[sNode][dNode])
		writeStr += " " + str(dNode) + " " + "{:.4f}".format(topicTopicGammaPriorParam[sNode][dNode])

	writeStr += "\n"

	ttPriorFile.write(writeStr)

ttPriorFile.close()
