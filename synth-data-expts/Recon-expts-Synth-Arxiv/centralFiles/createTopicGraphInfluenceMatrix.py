from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter

import networkx as nx

G = nx.erdos_renyi_graph(10,0.3)

ttGraphFile = open("topicTopicGraph.txt", "w")
adjList = defaultdict(lambda:defaultdict(int))


for line in nx.generate_adjlist(G):

	flds = line.strip().split()
	neighbors = len(flds) - 1

	if neighbors > 0:
		writeStr = str(neighbors) + " " + line.strip() + "\n"
		ttGraphFile.write(writeStr)

	src = int(flds[0].strip())

	for eachDestStr in flds[1:]:
		dest = int(eachDestStr.strip())
		adjList[src][dest] = 1

ttGraphFile.close()


topicTopicInfluenceMatrix = defaultdict(lambda: defaultdict(float))

# pwLenFile = open("topicTopicGraphPairwiseLength.txt", "w")

# allPairwiseDists = dict(nx.all_pairs_shortest_path_length(G))

# for sourceNode in allPairwiseDists:

# 	for destNode in allPairwiseDists[sourceNode]:

# 		dist = allPairwiseDists[sourceNode][destNode]
# 		writeStr = str(sourceNode) + " " + str(destNode) + " " + str(dist) + "\n"
# 		# pwLenFile.write(writeStr)

# 		kParam = 1
# 		thetaParam = 0.2/(dist + 1)
# 		topicTopicInfluenceMatrix[sourceNode][destNode] = np.random.gamma(kParam, thetaParam, 1)[0]


# # pwLenFile.close()
print adjList

for sNode in sorted(adjList.keys()):

	for dNode in sorted(adjList[sNode].keys()):

		# if adjList[sNode].get(dNode, -1) != -1:
		kParam = 5
		thetaParam = 0.2
		topicTopicInfluenceMatrix[sNode][dNode] = np.random.gamma(kParam, thetaParam, 1)[0]
			# topicTopicInfluenceMatrix[sNode][dNode] = thetaParam

		# else:
		# 	kParam = 1
		# 	thetaParam = 0.02
		# 	topicTopicInfluenceMatrix[sNode][dNode] = np.random.gamma(kParam, thetaParam, 1)[0]
			# topicTopicInfluenceMatrix[sNode][dNode] = thetaParam



print topicTopicInfluenceMatrix

ttPriorFile = open("topicTopicInfluenceFile.txt", "w")

for sNode in sorted(topicTopicInfluenceMatrix.keys()):
	writeStr = str(len(topicTopicInfluenceMatrix[sNode])) + " " + str(sNode)

	for dNode in sorted(topicTopicInfluenceMatrix[sNode].keys()):
		# writeStr += " " + str(dNode) + " " + str(topicTopicInfluenceMatrix[sNode][dNode])
		writeStr += " " + str(dNode) + " " + "{:.4f}".format(topicTopicInfluenceMatrix[sNode][dNode])

	writeStr += "\n"

	ttPriorFile.write(writeStr)

ttPriorFile.close()
