#include <iostream>				//for basic C++ functions 
#include <cstdio>				//for std C functions
#include <fstream>				//for i/o stream
#include <sstream>				//for parse data using stringstreams...
#include <string>				//for C++ string functions
#include <cstring>				//we need this for memset... and string functions from C
#include <unordered_map>		//to maintain the map of user-tweet_count, and user_user_tweet_count
#include <map>		//to maintain the map of user-tweet_count, and user_user_tweet_count
#include <ctime>				//for time realted functions... to convert user-formatted time to timestamp
#include <vector>
#include <assert.h>				//for assertions
#include <limits>		
#include <cmath>
// #include <math.h>
// #include <cdouble>
#include <algorithm>
#include <numeric>	
#include <random>
#include <chrono>
#include <cfenv>
#include <iomanip>


// #pragma STDC FENV_ACCESS ON

using namespace std;
using namespace std::chrono;


// using ull = unsigned long long int;
using ll = long long int;
using li = long int;
using ui = unsigned int;

#define INF std::numeric_limits<int>::max()					//use limits for infinity
#define PI 3.14159265


int BURN_IN = 199;
int ITERATIONS = 300;

int maxNumNodes = 151;
int numNodes = maxNumNodes;
int whichPrior = 1;
int vocabsize = 33000;
int numTopics = 25;
int maxNumTopics = numTopics;
int numMentions = 0;
int totalWords = 0;


double betaTTPrior = 0.01, gammaUTPrior = 0.01;
double sumBetaTTPrior = betaTTPrior*numTopics;
double sumGammaUTPrior = gammaUTPrior*numTopics;
double alphaTopicWordPrior = 0.01;
double sumAlphaTopicWordPrior = alphaTopicWordPrior * vocabsize;
double sigmaTopicMentionPrior = 0.01;
double sumSigmaTopicMentionPrior = sigmaTopicMentionPrior * numMentions;

float timeScalingFactor = 1.0;
// float timeScalingFactor = 960.0;

// alpha, beta gamma initialization for userUserInfluence...
double baseAlpha = 0.01, baseBeta = 1;

double logLikelihood = 0;


// function declaration

int getTopicTopicCombinations();
int printTopicTopicCount();

// Read data from files....
// vector < vector <double> > getSyntheticEventsFromFile(string fileName);
vector < vector <int> > getSyntheticEventsFromFile(string fileName);
vector < vector <int> > getEventsFromFile(string fileName);
vector < vector <ui> > getSyntheticDocsFromFile(string fileName);
// unordered_map <ll, vector <ll> > readIntVectorMapFromFile(string fileName);
vector < vector <ui> > readIntVectorMapFromFile(string fileName);
vector< vector <double> > readMultipleDoubleVectorsFromFile(string fileName);
vector <double>  readDoubleVectorFromFile(string fileName);
unordered_map <int, unordered_map <int, double> > readUserUserInfluenceFromFile(string fileName);
unordered_map <ui, unordered_map <ui, float> > getUserUserInfluence(string fileName);
map<ui, double> getUserBaseRates(string fileName);
int readHyperParametersVectorsFromFile();

// initialization
vector< vector <int> > initializeForSampler(vector< vector <int> > allEvents, int flag);
vector< vector <int> > initializeEvents(vector< vector <int> > allEvents, int flag);
int initializeTopics();
int initializeAllCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeBaseRateAndInfluence();
int initializeBaseRates();
int initializeTopicPopCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeUserUserInfluence();
int initializeAndUpdateAlphaValue();
int updateUserBaseRates();
int initializeAvgProbabilityVectors();
int initializeAvgTopicProbabilityVectors();

// sample topic
int sampleTopicAssignment(int ITE);
int getSampledTopicAssignment(int eventIndex, int eventNode, int eventParent, vector<ui> doc, int ITE);
double getFirstTermOfTopicAssignmentCondProb(int eventNode, int eventParent, int topic);
double getMiddleTermOfTopicAssignmentCondProb(int topic, unordered_map <int, ui> childEventTopicsHist, int eventIndex, int eventParent);
double getThirdTermOfTopicAssignmentCondProb(int topic, vector<ui> doc, vector<ui> wordHistVec);
double getTopicWordTerm(int topic, vector<ui> doc, vector<ui> wordHistVec);
unordered_map <int, ui> getHistOfTopicsOverChildEvents(int eventIndex);
unordered_map <ui, ui> getHistOfWordsOverWordsFromDoc(vector<ui> doc);
int createWordHistForAllDocs();


// sample parent
int sampleParentAssignment(int ITE);
int getSampledParentAssignment(double eventTime, int eventNode, int eventIndex, int eventTopic, int ITE);
vector<double> populateCalculatedProbVec(vector <ui> possibleParentEvents, vector<double> possibleParentExp, int eventNode, int eventTopic, double eventTime, int ITE);
double getFirstTermOfParentAssignment(int possParentEventTopic, int eventTopic);
double getFirstTermOfParentAssignmentNoParent(int eventNode, int eventTopic);
vector<ui> getPossibleParentEvents(int eventNode, int eventIndex);
vector<double> computeParentExponentials(vector<ui> possibleParentEvents, int eventIndex);


// sample Influence
int sampleInfluenceAssignment(int ITE);
int updateNodeNodeCountMap();


// decreament
int decreamentCountFromMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> doc, bool topicSampling);
int decreamentCountFromTopicTopic(int eventParentTopic, int eventTopic);
int decreamentCountFromUserTopic(int eventNode, int eventTopic);
int decreamentCountFromTopicWord(int eventTopic, vector <ui> doc);
int decreamentCountsFromChildEvents(int eventTopic, int eventIndex);


// increament
int increamentCountToMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector<ui> doc, bool topicSampling);
int increamentCountInTopicTopic(int eventParentTopic, int eventTopic);
int increamentCountInUserTopic(int eventNode, int eventTopic);
int increamentCountInTopicWord(int eventTopic, vector<ui> doc);
int increamentCountsForChildEvents(int eventTopic, int eventIndex);

// validations
int countCorrectFractionTopicAssignments();
int countCorrectFractionParentAssignments();
int writeEvery10ItersParentAssignmentToFile();
int writeEvery10ItersTopicAssignmentToFile();
int writeTopicPopularityCount();
int writeAvgProbVectorsToFile();
int writeAvgTopicProbVectorsToFile();
int writeTopicTopicInteractionToFile();
int writeUserUserInfluenceToFile();
int writeUserBaseRatesToFile();
int writeNodeNodeCountAndNodeCount();

// util functions
int getCellKey(int firstPart, int secondPart);
int getSampleFromMultinomial(vector<double> calculatedProbVec);
vector<double> getNormalizedLogProb(vector<double> calculatedProbVec);
int getSampleFromDiscreteDist(vector<double> normalizedProbVector);
double getSampleFromGamma(double alpha, double beta);
int printUnorderedMap(string matrixName);
string& SSS (const char* s);


int getAvgChildEvents();
int populateParentEventsForAll();
int populateParentEventsForAllFromFile(string parentsFile, string parentExpFile);
int writeAssignmentsToFile();

unordered_map <string, string> getConfigInputOutputFileNames(string fileName);

int printTopicTopicCorrelation();

bool sortcol(const vector<float>& v1, const vector<float>& v2)
{
	// return v1[1] < v2[1];
	return v1[1] > v2[1]; 				// sort descending...
}


// global members declaration
vector< vector <int> > allEvents;
vector< vector <int> > newSyntheticEvents;
vector< vector <ui> > allEventsDocs;

// unordered_map <int, unordered_map<int, double> > origUserUserInfluence;
// unordered_map <int, unordered_map<int, double> > userUserInfluence;

vector<double> hyperBeta, hyperGamma, hyperAlpha;
double sumAlpha, sumBeta, sumGamma;
vector< vector <double> > topicTopicProbVector,  userTopicPrefVector, topicWordProbVector;

// to maintain a map of eventIndex and docSize...
unordered_map <ui, int> eventIndexDocSizeMap;

map<ui, vector<ui> > nodeEventsMap;

// double sumAlpha;

// Count Matrices... 
// std::vector<std::vector<int>> vec_2d(rows, std::vector<int>(cols, 0));
vector < vector < ui > > NTWCountVec(numTopics, vector < ui >(vocabsize, 0));
vector < vector < ui > > NTTCountVec(numTopics, vector < ui >(numTopics, 0));
vector < vector < ui > > NUTCountVec(maxNumNodes, vector < ui >(numTopics, 0));

vector < ui > NTWSumWordsVec(numTopics);
vector < ui > NTTSumTopicsVec(numTopics);
vector < ui > NUTSumTopicsVec(numNodes);

vector < vector <ui> > wordHistAllDocsVector;

// unordered_map <int, unordered_map<int, double> > origUserUserInfluence;
unordered_map <ui, unordered_map<ui, double> > userUserInfluence;
unordered_map <ui, unordered_map<ui, double> > userUserInfEvery10thIter;
// vector < vector <ll> > followersMap(maxNumNodes, vector < ll >(maxDegree + 1, -1) );
vector < vector <ui> > followersMap;
// vector < vector <ll> > reverseFollowersMap(maxNumNodes, vector < ll >(maxDegree + 1, -1) );
vector < vector <ui> > reverseFollowersMap;

vector < vector <ui> > userGraphMap;
vector < vector <ui> > reverseUserGraphMap;

vector <vector <ui> > allPossibleParentEvents;
vector <vector <double> > allPossibleParentEventsExponentials;
vector <vector <double> > avgProbParForAllEvents;
vector <vector <double> > avgTopicProbVector;


// unordered_map<ui, ui> nodeEventsCountMap;
vector<ui> nodeEventsCountMap(maxNumNodes, 0);
unordered_map<ui, unordered_map<ui, ui> > nodeNodeCount;
unordered_map<ui, double> eventIndexTimestamps;

// unordered_map <ui, vector<ui> > childEventsMap;
map <ui, vector<ui> > childEventsMap;
vector<ui> levelInfo;

unordered_map<ui, vector <ui> > nodeNodeCombUpdateInfluence;

// validation Maps
// unordered_map <int, vector<int> > topicDistEvery10thIter, parentDistEvery10thIter;
map <int, vector<int> > topicDistEvery10thIter, parentDistEvery10thIter;
unordered_map <string, string> configInputFiles, configOutputFiles;

int flag = 0;
int ofHitCount = 0;

double totalTime;

int maxTrainLevel = 3;
int percentLessCount = 70;

double defaultMuVal;
double defaultUserBaseRate;
map<ui, double> userBaseRateMap;

long int countInteractions = 0;

// double logLikelihood = 0;
double alphaEstimate = 0;

int avgOverNumChains = 0;

int invalidTopicId = 1000;
int invalidParId = -2;


double betaTopic = 0.1;
vector<int> topicPopularityCount (numTopics, 0);


int main(int argc, char *argv[])
{

	ofstream llFile;
	llFile.open("estAllLogLikelihood.txt");

	// Read all the events -- "tStamp uid -1 -1"
	allEvents = getEventsFromFile("../../centralFiles/events_scaledTime_EventSelect_0_testThresh_3_set_1.txt");
	// allEvents = getEventsFromFile("./centralFiles/events_10.txt");
	cout << "Got all the events --- " << allEvents.size() << endl;
	
	// Read all the docs...
	allEventsDocs = getSyntheticDocsFromFile("../../centralFiles/dataFile.txt");
	// allEventsDocs = getSyntheticDocsFromFile("./centralFiles/docs_10.txt");
	cout << "Got all the docs  --- " << allEventsDocs.size() << endl;

	

	cout << "Creating hist of words for each doc...\n";
	createWordHistForAllDocs();
	cout << "Created hist of words for each doc...\n";

	// Get the list of empty events, random topic and no parent and not a parent for these events...
	// cout << "Populate list of empty events..." << endl;
	// populateListOfEmptyEvents();
	// cout << "Got the list of empty events..." << emptyEvents.size() << endl;


	readHyperParametersVectorsFromFile();
	cout << "updated hyperparameters...\n";

	// newSyntheticEvents = initializeForSampler(allEvents, flag);
	newSyntheticEvents = initializeEvents(allEvents, flag);
	initializeTopics();
	initializeAllCounts(newSyntheticEvents);

	cout << "done with initialization ... \n";

	
	for(int ITE = 0; ITE < ITERATIONS; ITE++)
	{
		logLikelihood = 0;

		high_resolution_clock::time_point t1, t2;
		auto duration = 0;
		t1 = high_resolution_clock::now();
		sampleTopicAssignment(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "sampled topics for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;
	
		llFile << setprecision(10) << logLikelihood << "\n";

		cout << "ITE -- " << ITE << endl;
		
		if(ITE % 10 == 0)
		{
			cout << "ITE -- " << ITE << endl;
			cout << "avg over num chains " << avgOverNumChains << endl;
		}
	}	// ITERATIONS...

	cout << "Done... Will be writing results to files...\n";

	// cout << "OverFlow Hit Count = " << ofHitCount << endl;
	llFile.close();

	writeEvery10ItersTopicAssignmentToFile();
	writeTopicPopularityCount();

	cout << "Avg over num chains : " << avgOverNumChains << endl;

	return 0;
}

///////////////////////////////
///////// INITIALIZATION //////
///////////////////////////////

vector< vector <int> > initializeEvents(vector< vector <int> > allEvents, int flag)
{
	// Hiding the topics... just an initialization...
	vector< vector <int> > localNewSyntheticEvents;
	// ll uNode, vNode;
	// int eventIndex = 0;
	int possParentEventId;

	// might have random child event map here...

	for(ui i = 0; i < allEvents.size(); i++)
	{
		vector<int> tempEvent;

		int currentEventNode = allEvents[i][1];

		// parent assignment initialization... 
		// if(allPossibleParentEvents[i].size() > 0 && emptyEvents[i] != 1)
		possParentEventId = 0;
		// possParentEventId = allEvents[i][2];
		int eventTopic = i % numTopics;
		// int eventTopic = allEvents[i][3];

		tempEvent.push_back(i);												// event Id
		tempEvent.push_back(currentEventNode);								// event User
		tempEvent.push_back(possParentEventId);								// event Parent
		tempEvent.push_back(eventTopic);									// event Topic
		tempEvent.push_back(allEvents[i][4]);								// event Level

		localNewSyntheticEvents.push_back(tempEvent);
	}

	return localNewSyntheticEvents;
}

int initializeTopics()
{
	newSyntheticEvents[0][3] = 0;
	vector<ui> currDoc = allEventsDocs[0];
	increamentCountInTopicWord(newSyntheticEvents[0][3], currDoc);
	// topicDistEvery10thIter[0].push_back(newSyntheticEvents[0][3]);

	// cout << 0 << " " << 0 << " " << NTWSumWordsVec[0] << "\n";

	for(ui i = 1; i < newSyntheticEvents.size(); i++)
	{
		int eventIndex = i;
		int eLevel = newSyntheticEvents[i][4];

		if(eLevel >= 0 && eLevel <= maxTrainLevel)
		{
			currDoc = allEventsDocs[eventIndex];

			vector<double> topicAssignmentScores;

			vector<ui> wordHistVec = wordHistAllDocsVector[eventIndex];

			for(int topicId = 0; topicId < numTopics; topicId++)
			{
				double topicWordTerm = getTopicWordTerm(topicId, currDoc, wordHistVec);
				topicAssignmentScores.push_back(topicWordTerm);
			}

			int assignedTopicInd = getSampleFromMultinomial(topicAssignmentScores);
			increamentCountInTopicWord(assignedTopicInd, currDoc);

			newSyntheticEvents[i][3] = assignedTopicInd;
		}
		else
		{
			newSyntheticEvents[i][3] = invalidTopicId;
		}
	}

	int totalCount = 0;

	cout << "Coming here...\n";

	for(int i = 0; i < numTopics; i++)
	{
		cout << i << " " << NTWSumWordsVec[i] << endl;
		totalCount += NTWSumWordsVec[i];
	}
	cout << " Total Words after initialization : " << totalCount << endl;

	return 0;
}

int initializeAllCounts(vector< vector <int> > localNewSyntheticEvents)
{
	cout << "Initializing the topic-xxx counts..." << endl;

	initializeTopicPopCounts(localNewSyntheticEvents);

	return 0;
}

int initializeTopicPopCounts(vector< vector <int> > localNewSyntheticEvents)
{
	// initialization of the NTT Matrix... because of some initialization of parents...
	int sumAll = 0;

	for(ui i = 0; i < localNewSyntheticEvents.size(); i++)
	{
		int eNode = localNewSyntheticEvents[i][1];
		int eParent = localNewSyntheticEvents[i][2];
		int eTopic = localNewSyntheticEvents[i][3];
		int eLevel = localNewSyntheticEvents[i][4];

		if(eLevel >= 0 &&  eLevel <= maxTrainLevel)
		{
			topicPopularityCount[eTopic] += 1;
		}
	}

	return 0;
}



///////////////////////////////
///////  SAMPLE TOPIC    //////
///////////////////////////////

int sampleTopicAssignment(int ITE)
{
	cout << "Sample Topic Assignment\n";

	for(unsigned int i = 0; i < newSyntheticEvents.size(); i++)	
	{
		int assignedTopic;
		int eventIndex = newSyntheticEvents[i][0];
		int eventNode = newSyntheticEvents[i][1];
		int eventParent = newSyntheticEvents[i][2];				// should be -2 for the first iteration...
		int eventTopic = newSyntheticEvents[i][3];				// should be -2 for the first iteration...	
		int eventLevel = newSyntheticEvents[i][4];
		
		if(eventLevel >= 0 && eventLevel <= maxTrainLevel)
		{
			vector<ui> doc = allEventsDocs[i];
			
			topicPopularityCount[eventTopic] = topicPopularityCount[eventTopic] - 1;
			decreamentCountFromMatrices(eventIndex, eventNode, eventParent, eventTopic, doc, 1);
			
			assignedTopic =  getSampledTopicAssignment(eventIndex, eventNode, eventParent, doc, ITE);
			
			topicPopularityCount[assignedTopic] = topicPopularityCount[assignedTopic] + 1;
			increamentCountToMatrices(eventIndex, eventNode, eventParent, assignedTopic, doc, 1);

			newSyntheticEvents[eventIndex][3] = assignedTopic;
		}
		else
		{
			assignedTopic = invalidTopicId;
			newSyntheticEvents[eventIndex][3] = assignedTopic;
		}

		if(ITE > BURN_IN && ITE%10 == 0)
		{
			topicDistEvery10thIter[eventIndex].push_back(assignedTopic);
		}
	}

	// printTopicTopicCount();
	cout << "Done sampling topics... \n";
	return 0;
}


int getSampledTopicAssignment(int eventIndex, int eventNode, int eventParent, vector<ui> doc, int ITE)
{
	int assignedTopic = -1;

	vector <double> calculatedProbVec (numTopics, 0.0);

	for(int topic = 0; topic < numTopics; topic++)
	{
		// cout << "Topic iterator at - " << topic << endl;
		double firstTerm = 0.0;
		double thirdTerm = 0.0;

		firstTerm = log(betaTopic + topicPopularityCount[topic]);

		vector<ui> wordHistVec = wordHistAllDocsVector[eventIndex];

		thirdTerm = getTopicWordTerm(topic, doc, wordHistVec);
		// thirdTerm = 0;
		calculatedProbVec[topic] = firstTerm + thirdTerm;
		// cout << "Topic iterator at done -- " << topic << endl;
	}

	assignedTopic = getSampleFromMultinomial(calculatedProbVec);


	calculatedProbVec.clear();
	// cout << assignedTopic << "\n";

	return assignedTopic;
}


// this is required for calculating the middleterm of the propobability for topic assignment...
unordered_map <int, ui> getHistOfTopicsOverChildEvents(int eventIndex)
{
	unordered_map <int, ui> topicCounts;

	vector <ui> childEventsList;
	try
	{
		childEventsList =  childEventsMap.at(eventIndex);
	}
	catch(exception &e)
	{
		return topicCounts;							// returning empty dict/map
	}

	if(childEventsList.size() > 0)
	{
		for(unsigned int i = 0; i < childEventsList.size(); i++)
		{
			ui childEventIndex = childEventsList[i];
			int childEventTopic = newSyntheticEvents[childEventIndex][3];

			topicCounts[childEventTopic] += 1;
		}
	}

	return topicCounts;
}  


// double getThirdTermOfTopicAssignmentCondProb(int topic, vector<ui> doc, vector<ui> wordHistVec)
double getTopicWordTerm(int topic, vector<ui> doc, vector<ui> wordHistVec)
{
	double finalThirdValue = 1.0;
	double finalThirdValueNume = 1.0;
	double finalThirdValueDenom = 1.0;

	unsigned i = 0;
	while(i < wordHistVec.size())
	{
		ui word = wordHistVec[i];
		i++;
		ui wordCount = wordHistVec[i];

		// as for all the words this is same... we do not really need hyperAlpha right now...
		// double baseterm = alphaWord + NTWCountVec[topic][word];
		double baseterm = alphaTopicWordPrior + NTWCountVec[topic][word];

		long double valueForWord = 1.0;

		for(ui j = 0; j < wordCount; j++)
		{
			valueForWord *= (baseterm + j);
		}

		finalThirdValueNume *= valueForWord;
		i++;
	}

	double baseTermDenom = sumAlpha + NTWSumWordsVec[topic];

	for(ui i = 0; i < doc.size(); i++)
	{
		finalThirdValueDenom *= (baseTermDenom + i);
	}

	finalThirdValue = finalThirdValueNume / finalThirdValueDenom;

	return log(finalThirdValue);
}

// this is required for calculating the lastterm of the probability for topic assignment...
int createWordHistForAllDocs()
{
	cout << "I am coming here to create word hist...\n";
	for(unsigned int i = 0; i < allEventsDocs.size(); i++)
	{
		vector<ui> doc = allEventsDocs[i];

		unordered_map<ui, ui> wordHist;
		unordered_map<ui, ui>::iterator wordHistIt;
		wordHist = getHistOfWordsOverWordsFromDoc(doc);

		// wordHistAllDocs[i] = wordHist;
		vector <ui> tempWordCounts;
		for(wordHistIt = wordHist.begin(); wordHistIt != wordHist.end(); wordHistIt++)
		{
			tempWordCounts.push_back(wordHistIt->first);
			tempWordCounts.push_back(wordHistIt->second);
		}
		wordHistAllDocsVector.push_back(tempWordCounts);
	}

	return 0;
}

// this is required for calculating the lastterm of the probability for topic assignment...
unordered_map <ui, ui> getHistOfWordsOverWordsFromDoc(vector<ui> doc)
{
	unordered_map <ui, ui> wordCounts;

	for(unsigned int i = 0; i < doc.size(); i++)
	{
		int word = doc[i];

		wordCounts[word] += 1;
	}

	return wordCounts;
}

// Decreament Counters
int decreamentCountFromMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> doc, bool topicSampling)
{
	// cout << "decreament count from topic-topic matrix.. from the cell parentTopic -> eventTopic";
	// decreament only if the eventTopic is valid...
	if(eventTopic >= 0)
	{

		if(topicSampling == 1)
		{
			// decreament the count of words corresponding to this event...
			decreamentCountFromTopicWord(eventTopic, doc);

		}
	}

	return 0;
}


int decreamentCountFromTopicWord(int eventTopic, vector <ui> doc)
{
	if(NTWSumWordsVec[eventTopic] >= doc.size())
	{
		NTWSumWordsVec[eventTopic] -= doc.size();
	}
	else
	{
		// cout << "Somethings wrong with the Topic Word Counts... \n";
		NTWSumWordsVec[eventTopic] = 0;
	}

	for(unsigned int i = 0; i < doc.size(); i++)
	{
		if(NTWCountVec[eventTopic][doc[i]] > 0)
		{
			NTWCountVec[eventTopic][doc[i]] -= 1;
		}
	}

	return 0;
}


// Increament Counters
int increamentCountToMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector<ui> doc, bool topicSampling)
{
	// cout << "increament count to topic-topic matrix... in the cell parentTopic -> assignedTopic";

	if(topicSampling == 1)
	{
		// increament counts in topic-word...
		increamentCountInTopicWord(eventTopic, doc);
		// printUnorderedMap("psitopic");
	}

	return 0;
}


int increamentCountInTopicWord(int eventTopic, vector<ui> doc)
{
	NTWSumWordsVec[eventTopic] += doc.size();
	// NTSumWords[eventTopic] += doc.size();

	if(NTWSumWordsVec[eventTopic] > totalWords)
	{
		cout << "Total words = " << totalWords << " NTW sum = " << NTWSumWordsVec[eventTopic] << "\n";
		exit(0);
	}

	for(unsigned int i = 0; i < doc.size(); i++)
	{
		NTWCountVec[eventTopic][doc[i]] += 1;

		if(NTWCountVec[eventTopic][doc[i]] > totalWords)
		{
			cout << "Total words = " << totalWords << " NTW = " << NTWCountVec[eventTopic][doc[i]] << " " << doc[i] << "\n";
			exit(0);
		}
	}
	
	return 0;
}


///////////////////////////////
//////// UTIL FUNCTIONS ///////
///////////////////////////////


int getSampleFromMultinomial(vector<double> calculatedProbVec)
{
	int assignedInd;

	// get normalized prob vector...
	vector <double> normalizedProbVector = getNormalizedLogProb(calculatedProbVec);
	
	assignedInd = getSampleFromDiscreteDist(normalizedProbVector);

	logLikelihood += log(normalizedProbVector[assignedInd]);

    return assignedInd;
}


vector<double> getNormalizedLogProb(vector<double> calculatedProbVec)
{
	vector<double> normalizedProbVector(calculatedProbVec.size(), 0.0);

	// each of the terms will be normalized as:
	// log(pi) - ( log(mi) + log(sum (exp (log(pi) - log(mi)))) )

	double maxTerm = *max_element(calculatedProbVec.begin(), calculatedProbVec.end());

	double sumOverExp = 0.0;
	// sum(exp(log(pi) - log(mi)))
	for(unsigned int i = 0; i < calculatedProbVec.size(); i++)
	{
		sumOverExp += exp(calculatedProbVec[i] - maxTerm);
	}

	for(unsigned int i = 0; i < calculatedProbVec.size(); i++)
	{
		normalizedProbVector[i] = exp(calculatedProbVec[i] - (maxTerm + log(sumOverExp)));
		// normalizedProbVector[i] = exp(calculatedProbVec[i] - (log(sumOverExp)));
	}

	return normalizedProbVector;
}

int getSampleFromDiscreteDist(vector<double> normalizedProbVector)
{
	// remove .. later
	random_device rd;
	mt19937 gen(rd());
	
	// default_random_engine gen;

	discrete_distribution<int> distribution(normalizedProbVector.begin(), normalizedProbVector.end());
	int ind = distribution(gen);

	return ind;
}


///////////////////////////////
////////   VALIDATIONS   ///////
///////////////////////////////


int writeEvery10ItersTopicAssignmentToFile()
{
	ofstream topicAssignmentsFile;
	topicAssignmentsFile.open("topicAssignments.txt");
	// topicAssignmentsFile.open(configOutputFiles["topicAssignmentFile"]);

	cout << "Writing topic assingments to file...\n";

	map<int, vector <int> >::iterator topicDistEvery10thIterIterator;

	for(topicDistEvery10thIterIterator = topicDistEvery10thIter.begin(); topicDistEvery10thIterIterator != topicDistEvery10thIter.end(); topicDistEvery10thIterIterator++)
	{
		int eventId = topicDistEvery10thIterIterator->first;
		vector<int> assignments = topicDistEvery10thIterIterator->second;

		topicAssignmentsFile << eventId << " ";

		for(ui i = 0; i < assignments.size(); i++)
		{
			topicAssignmentsFile << assignments[i] << " ";			
		}

		topicAssignmentsFile << endl;
	}

	return 0;
}

int writeTopicPopularityCount()
{
	ofstream topicAssignmentsFile;
	topicAssignmentsFile.open("topicPopularity.txt");

	cout << "Writing topic popularity to file...\n";

	for(int i = 0; i < numTopics; i++)
	{
		topicAssignmentsFile << i << " " << topicPopularityCount[i] << "\n";		
	}

	topicAssignmentsFile.close();
	return 0;
}



///////////////////////////////
///////// READING DATA ////////
//////// AND PARAMETERS ///////
//////// FROM FILES ///////////
///////////////////////////////

// Reading input and preprocessing data...
vector < vector <int> > getEventsFromFile(string fileName)
{
	vector < vector <int> > localAllEvents;

	ifstream allEventsFile;
    allEventsFile.open(fileName);

    string line;
    stringstream ss;

    unsigned i = 0;

    // maxTrainLevel = 0;

    if(allEventsFile.is_open())
    {
    	while(getline(allEventsFile, line))
		{
			double eventTime;
			int eventNode, eventParent, eventTopic, eventLevel;
			// int level;

			ss.clear();
			ss.str("");

			ss << line;
			// ss >> eventTime >> eventNode >> eventParent >> eventTopic;
			ss >> eventTime >> eventNode >> eventParent >> eventTopic >> eventLevel;

			eventIndexTimestamps[i] = eventTime;

			vector<int> tempEvent;
			tempEvent.push_back(i);						// eventIndex...
			tempEvent.push_back(eventNode);
			tempEvent.push_back(eventParent);
			tempEvent.push_back(eventTopic);
			tempEvent.push_back(eventLevel);

			localAllEvents.push_back(tempEvent);

			line.clear();

			i++;

			tempEvent.clear();
		}
    }
	else
	{
		cout << "Error opening file -- " << fileName << "\n";
	}
	cout << "read " << localAllEvents.size() << " events\n";
	double firstTime = eventIndexTimestamps[0];
	double lastTime = eventIndexTimestamps[eventIndexTimestamps.size() - 1];

	cout << "First event time -- " << firstTime << "\n";
	cout << "Last event time -- " << lastTime << "\n";

	totalTime = lastTime - firstTime;

	allEventsFile.close();
	return localAllEvents;
}


vector < vector <ui> > getSyntheticDocsFromFile(string fileName)
{
	vector < vector <ui> > allDocs;

	double avgDocLength = 0;

	ifstream allDocsFile;
    allDocsFile.open(fileName);

    string line;
    stringstream ss;
    int i = 0;

    if(allDocsFile.is_open())
    {
	    while(getline(allDocsFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;

			int word;
			vector <ui> currDoc;
			while(ss >> word)
			{
				// vector<double> tempEvent;
				currDoc.push_back(word);
			}
			
			eventIndexDocSizeMap[i] = currDoc.size();
			i++;

			allDocs.push_back(currDoc);
			line.clear();

			avgDocLength += currDoc.size();

			// if(i > 100000)
				// break;
		}
		allDocsFile.close();

		totalWords = avgDocLength;
		cout << "Total Number of words = " << totalWords << "\n";

		avgDocLength = avgDocLength/allDocs.size();
		
		cout << "Read " << allDocs.size() << " documents\n";
		cout << "Avg Doc Length = " << avgDocLength << endl;
    }
    else
    {
    	cout << "Error opening file -- " << fileName << "\n";
    }
    cout << "read " << allDocs.size() << " docs\n";
	return allDocs;
}


int readHyperParametersVectorsFromFile()
{
	// hyperBeta = readDoubleVectorFromFile("hyperBetaValue.txt");
	// hyperGamma = readDoubleVectorFromFile("hyperGammaValue.txt");
	// hyperAlpha = readDoubleVectorFromFile("hyperAlphaValue.txt");


	// topicTopicProbVector = readMultipleDoubleVectorsFromFile("topicTopicProbVectors.txt");
	// userTopicPrefVector = readMultipleDoubleVectorsFromFile("nodeTopicProbVectors.txt");
	// topicWordProbVector = readMultipleDoubleVectorsFromFile("topicWordProbVectors.txt");

	vector<double> tempBeta(numTopics, 0.01);
	hyperBeta = tempBeta;

	vector<double> tempGamma(numTopics, 0.01);
	hyperGamma = tempGamma;

	vector<double> tempAlpha(vocabsize, alphaTopicWordPrior);
	hyperAlpha = tempAlpha;

	sumAlpha = accumulate(hyperAlpha.begin(), hyperAlpha.end(), 0.0);
	sumBeta = accumulate(hyperBeta.begin(), hyperBeta.end(), 0.0);
	sumGamma = accumulate(hyperGamma.begin(), hyperGamma.end(), 0.0);
	return 0;
}
