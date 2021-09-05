# generate synthetic Wuv for given graph

from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

alphaAllEdges = 1.5
betaAllEdges = 0.2

f = open("uuGraph.txt")

wuvWriteFile = open("userUserInfluence.txt", "w")

i = 0
for line in f:
	flds = line.strip().split()
	count = int(flds[0].strip())
	uid = int(flds[1].strip())

	uidFoll = []
	wuvList = []

	numNeighs = len(flds) - 2

	wstr = str(numNeighs) + " " + str(uid) + " "

	j = 2
	while j < len(flds):

		vNode = int(flds[j].strip())
		wuvVal = np.random.gamma(alphaAllEdges, betaAllEdges, 1)[0]

		wuvList.append(vNode)
		wuvList.append(wuvVal)

		uidFoll.append(vNode)
		j += 1

	wstr += " ".join([str(x) for x in wuvList]) + "\n"

	wuvWriteFile.write(wstr)
	# tempFollowers[uid] = list(uidFoll)

	# print(i)
	# i += 1

f.close()

wuvWriteFile.close()
