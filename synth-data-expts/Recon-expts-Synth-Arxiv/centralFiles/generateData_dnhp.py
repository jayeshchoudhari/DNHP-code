from __future__ import division
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

#################### Global Part ###########################

Tmax = int(sys.argv[1])
alphaMultiplier = float(sys.argv[2])
seedVal = int(sys.argv[3])
np.random.seed(seedVal)

# Tmax = 100
# alphaMultiplier = 1
# seedVal = 1234
# np.random.seed(seedVal)
eventsToBeGenerated = 100000
maxLevelsToBeGenerated = 100

# Output Files...
# eventsFileName = "events_semisyn_sample_" + str(alphaMultiplier) + "_Tmax_" + str(Tmax) + "_seed_" + str(seedVal) + "_10K.txt"
# eventsFileName = "test_level_events_semisyn_sample.txt"
eventsFileName = "events_1L.txt"
# docsFileName = "docs_semisyn_sample_" + str(alphaMultiplier) + "_Tmax_" + str(Tmax) + "_seed_" + str(seedVal) + "_10K.txt"
# docsFileName = "test_level_docs_semisyn_sample.txt"
docsFileName = "docs_1L.txt"

allSyntEventsFileName = eventsFileName;
allSyntDocsFileName = docsFileName

# Input Files...
# nodeListFileName = "nodeListFile_100_40.0.txt"
# nodeListFileName = "users.txt"
userBaseRatesFileName = "./userBaseRates.txt"
topicBaseRatesFileName = "./topicBaseRates.txt"

userUserInfluenceFileName = "./userUserInfluence.txt"
topicTopicInfluenceFileName = "topicTopicInfluenceFile.txt"

userTopicPrefVectorsFileName = "userTopicDistribution.txt"
wordDistTopicsFileName = "topicWordDistribution.txt"


numNodes = 50
numTopics = 10
vocabsize = 500
topics = [i for i in range(0,numTopics)]

docsize = 10
lenLambda = 10

rayleighSigma = 3

alphaArrivsWuv = defaultdict(list)
alphaArrivsInterArrivs = defaultdict(list)
meanWuvs = defaultdict(list)

eventsCount = 0

#################### Global Part ###########################

#################### Functions ########################

def get_lambda_for_level_l(startTime, x, W_uv):
	
	timeDiff = x - startTime
	# func_value = W_uv * math.exp(-timeDiff)
	func_value = W_uv * math.exp(- alphaMultiplier * timeDiff)

	return func_value


def get_lambda_for_level_l_rayleigh(startTime, x, W_uv):

	timeDiff = x - startTime
	# func_value = W_uv * math.exp(-timeDiff)
	xBySigmaSqr = timeDiff*1.0/(rayleighSigma**2)

	xSqrBy2SigmaSqr = (timeDiff**2)*1.0 / (2*(rayleighSigma**2))

	func_value = W_uv * xBySigmaSqr * math.exp(-xSqrBy2SigmaSqr)

	return func_value	


def get_nhpp_timestamps_for_level_l(Tmax, lambdaUpperBound, t_e, W_uv):
	# print Tmax
	T, t0, n, m = Tmax, t_e, 0, 0
	lambda_constant = lambdaUpperBound 
	sm = t_e
	# homoTime = []
	inhomoTime = []
	# allDs = []

	while sm < T:
		u = np.random.uniform(0,1)
		w = -(np.log(u) / lambda_constant)						# so that w ~ exp(lambda_constant)
		sm = sm + w 
		probVal = (get_lambda_for_level_l(t_e, sm, W_uv))/lambda_constant
        # probVal = (get_lambda_for_level_l_rayleigh(t_e, sm, W_uv))/lambda_constant
		if sm < T and probVal >= 1e-7:
			# homoTime.append(sm)									# sm are the points in the homo. PP
			d = np.random.uniform(0,1)						
			# print(sm, d, math.exp(-(sm - t_e)))
			if d <= probVal:
				# allDs.append(d)
				inhomoTime.append(sm)							# inhomoTime are the points in inhomo. PP
				# n += 1
			# m += 1

		else:
			break

	# print (inhomoTime)
	# plot_for_confirmation(homoTime, inhomoTime, allDs)
	return inhomoTime


def get_homogenous_pp_timestamps(Tmax, lambda_rate):

	T, t0 = Tmax, 0
	hppTimeStamps = []
	prev_ti = t0

	while prev_ti <= Tmax:
		u = np.random.uniform(0,1)
		x = prev_ti - (np.log(u)/lambda_rate)
		if x <= Tmax:
			hppTimeStamps.append(x)
		prev_ti = x

	return hppTimeStamps


def generateLevel0Events():

	global eventsCount

	noparent = -1
	localLevel0Events = []

	for node in usersThatEmit:

		for eachTopic in topicsThatEmit:

			userMuVal = userBaseRates[node]
			topicMuVal = topicBaseRates[eachTopic]

			totalBaseRate = userMuVal * topicMuVal

			superNodeL0Timestamps = get_homogenous_pp_timestamps(Tmax, totalBaseRate)

			for ts in superNodeL0Timestamps:

				localLevel0Events.append([ts, node, noparent, eachTopic, 0])
				eventsCount += 1

	return localLevel0Events


def generateLthLevelEvents(lMinus1thLevelEvents, levelVal):

	global eventsCount

	localCurrentLevelEvents = []

	print "Num events in prev level -- ", len(lMinus1thLevelEvents)

	for event in lMinus1thLevelEvents:

		u_ts = event[0]
		u = event[1]

		parentTopic = event[3]
		parentEventId = str(u) + "_" + str(u_ts)

		upresent = W.get(u, -1)

		if upresent != -1:

			for node in W[u]:

				kpresent = Tau.get(parentTopic, -1)

				if kpresent != -1: 

					for topic in Tau[parentTopic]:

						WuvTaukk = W[u][node] * Tau[parentTopic][topic]

						nhppTimestampsLevelL = get_nhpp_timestamps_for_level_l(Tmax, WuvTaukk, u_ts, WuvTaukk)

						for ts in nhppTimestampsLevelL:
							# probVec = np.array(topicTopicProbVectors[parentTopic])
							eta_n = topic

							localCurrentLevelEvents.append([ts, node, parentEventId, eta_n, levelVal])
							eventsCount += 1

							if eventsCount > eventsToBeGenerated:
								return localCurrentLevelEvents


	return localCurrentLevelEvents


def generate_synthetic_events():
	
	totalEventsCount = 0
	allEvents = []

	levelVal = 0
	level0Events = generateLevel0Events()

	print "Num of level 0 events: ", len(level0Events)

	# allEvents.append(level0Events)
	for eachEvent in level0Events:
		allEvents.append(eachEvent)

	totalEventsCount += len(level0Events)

	# prevLevelEvents = level0Events.copy()
	prevLevelEvents = level0Events[:]

	levelVal += 1

	while len(prevLevelEvents) > 0:

		print "Are u comin in..??"

		currLevelEvents = generateLthLevelEvents(prevLevelEvents, levelVal)
		totalEventsCount += len(currLevelEvents)

		for eachEvent in currLevelEvents:
			allEvents.append(eachEvent)
			
		prevLevelEvents[:] = []

		print "Total Event Counts = ", totalEventsCount, levelVal

		if totalEventsCount < eventsToBeGenerated:
			# prevLevelEvents = currLevelEvents.copy()
			prevLevelEvents = currLevelEvents[:]
			levelVal += 1

	# print "all level events : ", allEvents
	allEvents.sort(key = lambda row:row[0])

	# print allEvents

	uniqEventIdSortedIndex = {}
	for eid in range(len(allEvents)):
		ts = allEvents[eid][0]
		node = allEvents[eid][1]
		uniqEventId = str(node) + "_" + str(ts)
		uniqEventIdSortedIndex[uniqEventId] = eid

	for eid in range(len(allEvents)):
		uniqParentId = allEvents[eid][2]
		if uniqParentId != -1:
			indexOfParent = uniqEventIdSortedIndex[uniqParentId]
			allEvents[eid][2] = indexOfParent
		else:
			allEvents[eid][2] = -1

	return allEvents


def writeOnlyEventsToFile():

	print("Writing events to file....\n")

	allEventsFile = open(allSyntEventsFileName, 'w')
	# Sample topic for each event based on the parent
	for event in range(0, len(allSyntheticEvents)): 
        # allSyntheticEvents[event].append(-1)

		# write synthetic data to file...
		eventStr = ' '.join([str(ele) for ele in allSyntheticEvents[event]])
		eventStr = eventStr + "\n"
		allEventsFile.write(eventStr)

	allEventsFile.close()


# def generate_synthetic_docs(allSyntDocsFileName, allSyntheticEventsTopicsAssigned, vocabsize, numTopics, docsize):
def generate_synthetic_docs(allSyntDocsFileName, localAllSyntheticEvents):

	print("Generating Documents...")
	docsize = np.random.poisson(lenLambda)

	vocabulary = [i for i in range(0,vocabsize)]
	# Generate word distribution for each topic
	# hyperAlpha = [np.random.uniform(0,1) for k in range(0,vocabsize)]

	# Generate words for all the docs and write to file
	docWords = []

	allDocsFile = open(allSyntDocsFileName, 'w')

    # for eachdoc in allSyntheticEventsTopicsAssigned:
	for eachdoc in localAllSyntheticEvents:
		docsize = np.random.poisson(lenLambda)
		wordsInDoc = np.random.choice(vocabulary, docsize, True, wordDistTopics[eachdoc[3]])
		# draw doc size number of words
		docWords.append(wordsInDoc)
		docStr = ' '.join([str(word) for word in wordsInDoc])
		docStr = docStr + "\n"
		allDocsFile.write(docStr)
        # append event id

	allDocsFile.close()
	# print(docWords)
	return docWords


def getNodeList(fileName):

	nodesFile = open(fileName)

	tempNodeList = []

	for line in nodesFile:
		tempNodeList.append(int(line.strip()))

	return tempNodeList;


def getUserUserInfluence(fileName):

	wuvfile = open(fileName)

	tempWuv = {}

	for line in wuvfile:
		splitLine = line.split()
		count = int(splitLine[0].strip())
		uNode = int(splitLine[1].strip())

		tempWuv[uNode] = {}
		# followers[uNode] = []

		j = 2
		while j < len(splitLine):
			vNode = int(splitLine[j].strip())
			uvInf = float(splitLine[j+1].strip())
						
			tempWuv[uNode][vNode] = uvInf
			j = j + 2


	wuvfile.close()
	return tempWuv


def getFollowers(fileName):

	tempFollowers = {}

	f = open(fileName)

	i = 0
	for line in f:
		splts = line.strip().split()
		count = int(splts[0].strip())
		uid = int(splts[1].strip())

		uidFoll = []

		j = 2
		while j < len(splts):
			uidFoll.append(int(splts[j].strip()))
			j += 1


		tempFollowers[uid] = list(uidFoll)

		# print(i)
		# i += 1

	f.close()
	return tempFollowers


def getUserTopicPrefVectors(fileName):

	tempUserTopicPrefVector = {}

	f = open(fileName)

	for line in f:
		splts = line.strip().split()

		uid = int(splts[0].strip())

		tempVec = []
		sumVec = 0
		for j in range(1,len(splts)):
			ele = float(splts[j].strip())
			# sumVec += ele
			tempVec.append(ele)

		tempVec = np.array(tempVec)
		tempVec /= tempVec.sum()

		tempUserTopicPrefVector[uid] = list(tempVec)

	f.close()
	return tempUserTopicPrefVector


def getTopicWordProbVectors(fileName):

	tempWordDistVectors = []

	f = open(fileName)

	for line in f:
		splts = line.strip().split()

		tid = int(splts[0].strip())

		tempVec = []
		for j in range(1,len(splts)):
			tempVec.append(float(splts[j].strip()))

		tempVec = np.array(tempVec)
		tempVec /= tempVec.sum()

		tempWordDistVectors.append(list(tempVec))


	f.close()
	return tempWordDistVectors


def getUserBaseRates(fileName):

	tempUBR = {}

	f = open(fileName)

	for line in f:
		splts = line.strip().split()

		uid = int(splts[0].strip())
		ubr = float(splts[1].strip())

		tempUBR[uid] = ubr


	f.close()
	return tempUBR;


def getTopicTopicProbVectors(fileName):

	tempTopicTopicVec = []

	f = open(fileName)

	for line in f:
		splts = line.strip().split()

		topId = int(splts[0])
		topicVec = []

		j = 1
		while j < len(splts):
			topicVec.append(float(splts[j]))
			j = j + 1

		topicVec = np.array(topicVec)
		topicVec /= topicVec.sum()

		tempTopicTopicVec.append(list(topicVec))

	return tempTopicTopicVec

#################### Functions ###########################


############################## 
####### global part ##########
############################## 

# nodes = getNodeList(nodeListFileName)
# print("Got the list of nodes...", len(nodes))


'''
print("Getting followers map...")
followers = getFollowers(followersFileName)
print("Got the followers map...", len(followers)) 
'''

print("Getting user user influence map...")
W = getUserUserInfluence(userUserInfluenceFileName)
print("Got user-user influence matrix", len(W))

nodes = list(W.keys())
# numNodes = len(nodes)

print("Getting Topic-Topic influence map...")
Tau = getUserUserInfluence(topicTopicInfluenceFileName)
print("Got Topic-Topic influence matrix", len(Tau))

topics = list(Tau.keys())


print("Getting user base rates...")
userBaseRates = getUserBaseRates(userBaseRatesFileName)
print("Got user base rates", len(userBaseRates))

usersThatEmit = list(userBaseRates.keys())
print("users that emit -- ", len(usersThatEmit))


print("Getting topic base rates...")
topicBaseRates = getUserBaseRates(topicBaseRatesFileName)
print("Got user base rates", len(topicBaseRates))

topicsThatEmit = list(topicBaseRates.keys())
print("topics that emit -- ", len(topicsThatEmit))


'''
print("Getting Topic Topic Vectors...")
# topicTopicProbVectors = getTopicTopicProbVectors(topicTopicProbVectorsFileName)

topicTopicVecFile = open("topicTopicInteraction_dirichlet.txt", "w")
hyperBeta = [0.01]*numTopics
topicTopicProbVectors = []
for x in range(numTopics):
	np.random.seed()
	
	localTopicTopicVec = list(np.random.dirichlet(hyperBeta))
	topicTopicProbVectors.append(localTopicTopicVec)

	writeStr = str(x) + ' ' + ' '.join([str(i) for i in localTopicTopicVec])

	topicTopicVecFile.write(writeStr + "\n")

topicTopicVecFile.close()

print("Got topic-topic prob vectors", len(topicTopicProbVectors))
'''

####################################

print("Getting word dist for each topic")
# wordDistTopics = getTopicWordProbVectors(wordDistTopicsFileName)

wordDistTopics = []
hyperAlpha = [0.01]*vocabsize
'''
hyperAlpha = []

for i in range(numTopics):
	
	tempDist = [0]*vocabsize
	tempDist[i] = 2.0
	tempDist[i+numTopics] = 2.0

	wordList = [i, i+numTopics]

	for j in range(vocabsize):
		if j not in wordList:
			tempDist[j] = 0.0001

	hyperAlpha.append(tempDist)

# print hyperAlpha
'''
wordDistFile = open("topicWordDistribution_dirichlet.txt", "w")
for x in range(numTopics):
	np.random.seed()
	localWordDist = list(np.random.dirichlet(hyperAlpha))
	# localWordDist = list(np.random.dirichlet(hyperAlpha[x]))
	wordDistTopics.append(localWordDist)

	writeStr = str(x) + ' ' + ' '.join([str(i) for i in localWordDist]) 

	wordDistFile.write(writeStr + "\n")

wordDistFile.close()

print("Got word dist for each topic", len(wordDistTopics))


##########################

'''
print("Getting topic dist for each user")
# userTopicPrefVectors = getUserTopicPrefVectors(userTopicPrefVectorsFileName)

hyperGamma = [0.01]*numTopics
userTopicPrefVectors = defaultdict(list)
userTopicPrefFile = open("userTopicPreference_dirichlet.txt", "w")

for x in usersThatEmit:
	np.random.seed()
	
	localUserTopicPref = list(np.random.dirichlet(hyperGamma))
	userTopicPrefVectors[x] = localUserTopicPref

	writeStr = str(x) + ' ' + ' '.join([str(i) for i in localUserTopicPref]) 

	userTopicPrefFile.write(writeStr + "\n")

userTopicPrefFile.close()

print("Got user topic preferences", len(userTopicPrefVectors))
'''

# print("Read data from file")
print("Generating events...")
allSyntheticEvents = generate_synthetic_events()
print(len(allSyntheticEvents))

writeOnlyEventsToFile()

allSyntheticDocs = generate_synthetic_docs(allSyntDocsFileName, allSyntheticEvents)