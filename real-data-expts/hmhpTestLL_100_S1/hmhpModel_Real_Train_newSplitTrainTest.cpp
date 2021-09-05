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
int initializeNNTTCounts(vector< vector <int> > localNewSyntheticEvents);
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
	
	// Read all the mentions...
	// allEventsMentions = getSyntheticDocsFromFile("./centralFiles/dataMentionsFileTrain.txt");
	// cout << "Got all the Mention docs  --- " << allEventsMentions.size() << endl;

	// Get one entity per tweet for topic assignment (at the time of initialization)
	// entityPerTweet = getEntityPerTweet("./entitiesPerTweet.txt");
	// getEntityPerTweet("./centralFiles/entitiesPerTweetTrain.txt");
	// cout << "Got entity per tweet ---- " << entityPerTweet.size() << endl;

	cout << "Getting followers map\n";
	// followersMap = readIntVectorMapFromFile("../../../followers_map.txt");			// this would be required for the Wuv matrix.... 
	userGraphMap = readIntVectorMapFromFile("../../centralFiles/uuGraph.txt");
	cout << "Got the followers map... " << userGraphMap.size() << "\n";

	cout << "Creating hist of words for each doc...\n";
	createWordHistForAllDocs();
	cout << "Created hist of words for each doc...\n";

	// cout << "Creating hist of mentions for each doc...\n";
	// createMentionHistForAllDocs();
	// cout << "Created hist of mentions for each doc...\n";

	// Read the possible parent and parent exp values...
	cout << "Getting possible parent events for each event..." << endl;
	populateParentEventsForAllFromFile("../../centralFiles/top100Parents.txt", "../../centralFiles/top100ParentExps.txt");
	cout << "Got possible parent events -- " << allPossibleParentEvents.size() << " " << allPossibleParentEventsExponentials.size() << "\n";

	// Get the list of empty events, random topic and no parent and not a parent for these events...
	// cout << "Populate list of empty events..." << endl;
	// populateListOfEmptyEvents();
	// cout << "Got the list of empty events..." << emptyEvents.size() << endl;


	readHyperParametersVectorsFromFile();

	// newSyntheticEvents = initializeForSampler(allEvents, flag);
	newSyntheticEvents = initializeEvents(allEvents, flag);
	initializeTopics();
	initializeAllCounts(newSyntheticEvents);
	initializeBaseRateAndInfluence();
	initializeAvgProbabilityVectors();

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

		
		t1 = high_resolution_clock::now();
		sampleParentAssignment(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "sampled parents for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;
		
		// updating the childEventsMap... as it might change after each iteration once we incorporate parent sampling
		t1 = high_resolution_clock::now();

		childEventsMap.clear();
		for(unsigned int i = 0; i < newSyntheticEvents.size(); i++)
		{
			int eventCurrParent = newSyntheticEvents[i][2];

			if(newSyntheticEvents[i][4] >= 0 && newSyntheticEvents[i][4] <= maxTrainLevel)
			{
				if(eventCurrParent != -1)
				{
					childEventsMap[eventCurrParent].push_back(i);
				}
			}
		}

		updateNodeNodeCountMap();
	
		t2 = high_resolution_clock::now();
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		// cout << "Updating Child Events Map and Node-Node Count after -- "<< ITE << " Iteration --  Time  taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		sampleInfluenceAssignment(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		// cout << "Sampled User-User Influence for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		updateUserBaseRates();
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		// cout << "Updating Alpha Value and User Base Rates -- "<< ITE << "--  Time taken -- " << duration << endl;
        cout << "Default Mu Val = " << defaultMuVal << endl;

		// cout << "Log Likelihood = " << logLikelihood << "\n";
		// llFile << logLikelihood << "\n";

		// cout << setprecision(10) << "Log Likelihood = " << logLikelihood << "\n";
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
	writeEvery10ItersParentAssignmentToFile();
	// writeAvgProbVectorsToFile();
	// writeAvgTopicProbVectorsToFile();
	writeTopicTopicInteractionToFile();				// we will build this from avg probability parent assignment...
	writeUserUserInfluenceToFile();
	writeUserBaseRatesToFile();
	// writeNodeNodeCountAndNodeCount();

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
		
		int flag = 1;
		if(allPossibleParentEvents[i].size() > 0)
		{
			for(int j = 0; j < allPossibleParentEvents[i].size(); j++)
			{
				int candParId = allPossibleParentEvents[i][j];

				if(allEvents[candParId][4] >= 0 &&  allEvents[candParId][4] <= maxTrainLevel)
				{
					possParentEventId = allPossibleParentEvents[i][j];
					flag = 0;
					break;
				}
			}
			if(flag == 1)
			{
				possParentEventId = -1;
			}
			// cout << "Event Id: " << " " << i << " " << possParentEventId << endl;
		}
		else
		{
			possParentEventId = -1;
			// cout << "Event Id: " << " " << i << " " << -1 << endl;
		}
		
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
				
			// newSyntheticEvents[i][3] = i % numTopics;
			// topicDistEvery10thIter[i].push_back(newSyntheticEvents[i][3]);
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

	initializeNNTTCounts(localNewSyntheticEvents);

	return 0;
}

int initializeNNTTCounts(vector< vector <int> > localNewSyntheticEvents)
{
	// initialization of the NTT Matrix... because of some initialization of parents...
	int sumAll = 0;

	ui countNuFor = (int)(percentLessCount * localNewSyntheticEvents.size() * 0.01);

	for(ui i = 0; i < localNewSyntheticEvents.size(); i++)
	{
		int eNode = localNewSyntheticEvents[i][1];
		int eParent = localNewSyntheticEvents[i][2];
		int eTopic = localNewSyntheticEvents[i][3];
		int eLevel = localNewSyntheticEvents[i][4];

		if(eLevel >= 0 && eLevel < maxTrainLevel)
		// if(i < countNuFor)
		{
			// nodeTopicCountMap[eNode][eTopic] += 1;
			// topicNodeCountMap[eTopic][eNode] += 1;
			cout << i << " ";
			nodeEventsCountMap[eNode] += 1;
			// cout << "updated node node count for enode.. " << eNode << "..\n";
			// topicEventsCountMap[eTopic] += 1;
		}
		if(eLevel >= 0 &&  eLevel <= maxTrainLevel)
		{
			cout << i << " ";
			if(eParent > -1)
			{
				int eParentNode = localNewSyntheticEvents[eParent][1];
				int eParentTopic = localNewSyntheticEvents[eParent][3];

				// Topic Topic counts...
				increamentCountInTopicTopic(eParentTopic, eTopic);
				// cout << "incremented count in topic-topic.. " << eParentTopic << " " << eTopic << "..\n";
				// Node Node counts...
				nodeNodeCount[eParentNode][eNode] += 1;
				// cout << "incremented nodeNode count.. " << eParentNode << " " << eNode << "..\n";
				// Node Sum counts --  Do we really need the sum...?
				// nodeNodeCountSum[eParentNode] += 1;

				// childEventsMap[eParentNode].push_back(i);
				childEventsMap[eParent].push_back(i);
				// cout << "adding event to child event map of parent -- " << eParent << endl;

				sumAll++;
			}
			else
			{
				// NUTCountVec[eNode][eTopic] += 1;
				// NUTSumTopicsVec[eNode] += 1;
				increamentCountInUserTopic(eNode, eTopic);
				// cout << "incremented count in user-topic.. " << eNode << " " << eTopic << "..\n";

			}
		}
	}

	cout << "Number of Edges = " << sumAll << "\n";
	// cout << "Number of non-zero topic-topic maps -- " << NTTopicsSecLevelCount.size() << endl;
	// cout << "Number of non-zero topic-topic maps -- " << topicTopicCount.size() << endl;

	return 0;
}


int initializeBaseRateAndInfluence()
{
	initializeUserUserInfluence();
	// initializeTopicTopicInfluence();

	// initializeBaseRates();
	updateUserBaseRates();
	return 0;
}


int initializeUserUserInfluence()
{
	cout << "Initializing user user influence...\n";

	ui count = 0;
	ui uNode, vNode;
	int numTimesAddedNu = 0;

	double avgVal = 0.0;

	for(ui i = 0; i < userGraphMap.size(); i++)
	{
		vector <ui> followers = userGraphMap[i];
		// cout << "At Node -- " << i << endl;
		uNode = i;
		
		double scaleParam = 1/(nodeEventsCountMap[uNode] + baseBeta);
		
		unordered_map <ui, double> tempNodeInf;

		for(ui j = 0; j < followers.size(); j++)
		{
			vNode = followers[j];

			double shapeParam = nodeNodeCount[uNode][vNode] + baseAlpha;

			// double infVal  = shapeParam * scaleParam;
			double infVal = getSampleFromGamma(shapeParam, scaleParam);

			tempNodeInf[vNode] = infVal;

			// userUserInfluenceSumPerNode[uNode] += infVal;
			avgVal += infVal;

			// cout << uNode << " " << vNode << " " << nodeNodeCount[uNode][vNode] << " " << nodeEventsCountMap[uNode] << " " << infVal << endl;

			count++;
		}

		userUserInfluence[uNode] = tempNodeInf;
		tempNodeInf.clear();

	}

	cout << "Avg Wuv val = " << avgVal / count << endl; 

	cout << "Initialized user user influence -- " <<  count << " users...\n";
	return 0;
}

int initializeAvgProbabilityVectors()
{
	for(ui i = 0; i < allPossibleParentEvents.size(); i++)
	{
		int probVecSize = allPossibleParentEvents[i].size();
		vector<double> initVec(probVecSize + 1, 0.0);
		avgProbParForAllEvents.push_back(initVec);
		initVec.clear();
	}

	return 0;
}



///////////////////////////////
/////// User Base Rates  //////
///////////////////////////////


int updateUserBaseRates()
{
	int sponCount = 0;
	int minTime = eventIndexTimestamps[0];
	int maxTime = eventIndexTimestamps[eventIndexTimestamps.size() - 1];

	int totalDataObservedTime = maxTime - minTime;

	map<int, int> eachNodeSponCount;


	map<int, bool> distinctNodes;

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{	
		if(newSyntheticEvents[i][4] >= 0 &&  newSyntheticEvents[i][4] <= maxTrainLevel)
		{
			if(newSyntheticEvents[i][2] == -1)
			{
				sponCount++;
				distinctNodes[newSyntheticEvents[i][1]] = 1;
				eachNodeSponCount[newSyntheticEvents[i][1]]++;
			}
		}
	}

	// defaultMuVal = (sponCount * 1.0) / ((maxTime - minTime) * distinctNodes.size());

	double averageMuVal = 0;

	double minMuVal = 1000000;
	double maxMuVal = 0;
	map<int, int>::iterator eachNodeSponCountIt;	

	for(eachNodeSponCountIt = eachNodeSponCount.begin(); eachNodeSponCountIt != eachNodeSponCount.end(); eachNodeSponCountIt++)
	{
		int nodeId = eachNodeSponCountIt->first;
		int nodeSponCount = eachNodeSponCountIt->second;

		userBaseRateMap[nodeId] = (nodeSponCount * 1.0) / totalDataObservedTime;

		averageMuVal +=  userBaseRateMap[nodeId];

		if(userBaseRateMap[nodeId] > maxMuVal)
		{
			maxMuVal = userBaseRateMap[nodeId];
		}

		if(userBaseRateMap[nodeId] < minMuVal)
		{
			minMuVal = userBaseRateMap[nodeId];
		}
	}

	defaultMuVal = averageMuVal / eachNodeSponCount.size();
	defaultUserBaseRate = averageMuVal / eachNodeSponCount.size();
	
	cout << "sponCount = " << sponCount << " minMuVal = " << minMuVal << " maxMuVal = " << maxMuVal << " MLE MuVal = " << defaultMuVal << " " << defaultUserBaseRate << endl;

	return 0;
}


///////////////////////////////
/////// SAMPLE INFLUENCE //////
///////////////////////////////

int sampleInfluenceAssignment(int ITE)
{
	unordered_map<ui, unordered_map<ui, ui> >::iterator nodeNodeCountIterator;
	int count = 0;

	ui uNode, vNode;	
	// int numTimesAddedNu = 0;

	double avgVal = 0.0;

	for(ui i = 0; i < userGraphMap.size(); i++)
	{
		vector <ui> followers = userGraphMap[i];
		
		uNode = i;
		
		double scaleParam = 1/(nodeEventsCountMap[uNode] + baseBeta);
		
		unordered_map <ui, double> tempNodeInf;

		for(ui j = 0; j < followers.size(); j++)
		{
			vNode = followers[j];

			double shapeParam = nodeNodeCount[uNode][vNode] + baseAlpha;

			double wuv = getSampleFromGamma(shapeParam, scaleParam);

			userUserInfluence[uNode][vNode] = wuv;

			if(ITE%10 == 0 && ITE >= BURN_IN)
			{
				userUserInfEvery10thIter[uNode][vNode] += wuv;
			}

			avgVal += wuv;

			count += 1;

			// cout << uNode << " " << vNode << " " << nodeNodeCount[uNode][vNode] << " " << nodeEventsCountMap[uNode] << " " << wuv << endl;
		}

		nodeNodeCount[uNode].clear();
	}

	if(ITE%10 == 0 && ITE >= BURN_IN)
	{
		avgOverNumChains = avgOverNumChains + 1;
	}

	

	nodeNodeCount.clear();
	cout << "Avg Wuv value = " << (avgVal * 1.0) / count  << " Over Edges = " << count << endl;

	return 0;
}



int updateNodeNodeCountMap()
{
	int eventNode, parentEvent, parentNode, eventLevel;
	
	nodeNodeCount.clear();

	for(unsigned int i = 0; i < newSyntheticEvents.size(); i++)
	{
		eventNode = newSyntheticEvents[i][1];
		parentEvent = newSyntheticEvents[i][2];
		eventLevel = newSyntheticEvents[i][4];

		if(eventLevel >= 0  && eventLevel <= maxTrainLevel)
		{
			if(parentEvent >= 0)
			{
				parentNode = newSyntheticEvents[parentEvent][1];
				nodeNodeCount[parentNode][eventNode]++;
			}
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
		// int eventIndex = i;
		int eventNode = newSyntheticEvents[i][1];
		int eventParent = newSyntheticEvents[i][2];			// should be -2 for the first iteration...
		int eventTopic = newSyntheticEvents[i][3];				// should be -2 for the first iteration...	
		int eventLevel = newSyntheticEvents[i][4];
		
		if(eventLevel >= 0 && eventLevel <= maxTrainLevel)
		{
			vector<ui> doc = allEventsDocs[i];
			
			// cout << "Working for event -- " << i << endl;
			decreamentCountFromMatrices(eventIndex, eventNode, eventParent, eventTopic, doc, 1);
			// cout << "Working for event -- " << i << " decreament done " << endl;
			assignedTopic =  getSampledTopicAssignment(eventIndex, eventNode, eventParent, doc, ITE);
			// cout << "Working for event -- " << i << " got topic " << endl;
			increamentCountToMatrices(eventIndex, eventNode, eventParent, assignedTopic, doc, 1);
			// cout << "Working for event -- " << i << " increament done " << endl;

			newSyntheticEvents[eventIndex][3] = assignedTopic;

			// printf("%d %d\n", assignedTopic, i);
			
			// if(i % 10000 == 0)
				// printf("Topic Assignment -- %d\n", i);
		}
		else
		{
			assignedTopic = invalidTopicId;
			newSyntheticEvents[eventIndex][3] = assignedTopic;
		}

		if(ITE > BURN_IN && ITE%10 == 0)
		{
			topicDistEvery10thIter[eventIndex].push_back(assignedTopic);

			// if(i % 100000 == 0)
				// cout << i << endl;
		}
	}

	// printTopicTopicCount();
	cout << "Done sampling topics... \n";
	return 0;
}

/*
int printTopicTopicCount()
{
	int sumAll = 0;
	for(ui i = 0; i < numTopics; i++)
	{
		for(ui j = 0; j < numTopics; j++)
		{
			// cout << NTTCountVec[i][j] << " ";
			sumAll += NTTCountVec[i][j];
		}
		// cout << "\n";
	}
	cout << "Sum Over all i-j -- " << sumAll << "\n";
	
	if(sumAll > countInteractions)
	{
		cout << "Sum all is greater than Count Interactions (For first iteration it can be...)\n";
		cout << sumAll << " " << countInteractions << "\n";
		// exit(0);
	}

	return 0;
}
*/

int getSampledTopicAssignment(int eventIndex, int eventNode, int eventParent, vector<ui> doc, int ITE)
{
	int assignedTopic = -1;

	vector <double> calculatedProbVec (numTopics, 0.0);
	
	unordered_map <int, ui> childEventTopicsHist = getHistOfTopicsOverChildEvents(eventIndex);

	for(int topic = 0; topic < numTopics; topic++)
	{
		// cout << "Topic iterator at - " << topic << endl;
		double firstTerm = 0.0;
		double middleTerm = 0.0;
		double thirdTerm = 0.0;

		// cout << "eventIndex -- " << eventIndex << endl;
		firstTerm = getFirstTermOfTopicAssignmentCondProb(eventNode, eventParent, topic);
		
		if(childEventTopicsHist.size() > 0)
		{
			middleTerm = getMiddleTermOfTopicAssignmentCondProb(topic, childEventTopicsHist, eventIndex, eventParent);
		}
		else
		{
			middleTerm = 0;
		}
		// middleTerm = 0;
		
		vector<ui> wordHistVec = wordHistAllDocsVector[eventIndex];
		// thirdTerm = getThirdTermOfTopicAssignmentCondProb(topic, doc, wordHistVec);
		thirdTerm = getTopicWordTerm(topic, doc, wordHistVec);
		// thirdTerm = 0;

		calculatedProbVec[topic] = firstTerm + middleTerm + (thirdTerm);
		// cout << "Topic iterator at done -- " << topic << endl;
	}

	assignedTopic = getSampleFromMultinomial(calculatedProbVec);

	// cout << "Got assigned topic... " << assignedTopic << endl;

	// if(ITE > BURN_IN && ITE%10 == 0)
	// {
	// 	cout << "I am not reaching here...\n";
	// 	for(ui i = 0; i < avgTopicProbVector[eventIndex].size(); i++)
	// 	{
	// 		avgTopicProbVector[eventIndex][i] += calculatedProbVec[i];
	// 	}
	// }

	calculatedProbVec.clear();
	// cout << assignedTopic << "\n";

	return assignedTopic;
}

double getFirstTermOfTopicAssignmentCondProb(int eventNode, int eventParent, int topic)
{
	double finalFirstTerm = 0;

	if(eventParent >= 0)
	{
		int eventParentTopic = newSyntheticEvents[eventParent][3];
		// finalFirstTerm = hyperBeta[topic] + NTTCountVec[eventParentTopic][topic];
		finalFirstTerm = hyperBeta[topic] + NTTCountVec[eventParentTopic][topic];
	}
	else if(eventParent == -1)
	{
		// no parent
		finalFirstTerm = hyperGamma[topic] + NUTCountVec[eventNode][topic];
		// cout << eventNode << " " << topic << endl;
	}
	else
	{
		// if the parent is invalid or not assigned yet...
		// finalFirstTerm = log(hyperGamma[topic]);
		finalFirstTerm = hyperGamma[topic];
	}

	double logVal = log(finalFirstTerm);
	// cout << logVal << endl;
	return logVal;
}


double getMiddleTermOfTopicAssignmentCondProb(int topic, unordered_map <int, ui> childEventTopicsHist, int eventIndex, int eventParent)
{
	double finalMiddleValue = 0.0;
	double finalMiddleValueNume = 0.0;
	double finalMiddleValueDenom = 0.0;

	unordered_map <int, ui>::iterator childEventTopicsHistIt;

	// int childEventTopicsHistCount = 0;

	for(childEventTopicsHistIt = childEventTopicsHist.begin(); childEventTopicsHistIt != childEventTopicsHist.end(); childEventTopicsHistIt++)
	{
		int lPrime = childEventTopicsHistIt->first;

		if(lPrime > -1)
		{
			int lPrimeCountInChilds = childEventTopicsHistIt->second;

			// double baseTerm = betaLPrime + ttCountWithoutChildEvents;
			double baseTerm = hyperBeta[lPrime] + NTTCountVec[topic][lPrime];

			double valueForLPrime = 0.0;

			// evaluating sum( log ( baseterm + i) )

			for(int i = 0; i < lPrimeCountInChilds; i++)
			{
				valueForLPrime += log(baseTerm + i);
			}

			finalMiddleValueNume += valueForLPrime;
		}
	}


	double baseTermDenom = sumBeta + NTTSumTopicsVec[topic];

	ui numChildEvents = 0;

	vector <ui> cEvents;

	try
	{
		numChildEvents = childEventsMap[eventIndex].size();

		numChildEvents = numChildEvents - childEventTopicsHist[-1];

	}
	catch(exception &e)
	{
		numChildEvents = 0;
	}

	if(numChildEvents > 0)
	{
		for(ui i = 0; i < numChildEvents; i++)
		{
			finalMiddleValueDenom += log(baseTermDenom + i);
		}
	}
	else
	{
		finalMiddleValueDenom = 0;
	}

	if(finalMiddleValueDenom == 0.0 && finalMiddleValueNume != 0.0)
	{
		cout << "Some issue with the code...\n";
		exit(0);
	}

	finalMiddleValue = finalMiddleValueNume - finalMiddleValueDenom;

	return finalMiddleValue;
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

///////////////////////////////
///////  SAMPLE PARENT   //////
///////////////////////////////

int sampleParentAssignment(int ITE)
{
	cout << "Sampling Parent Assignment ... \n";

	nodeNodeCombUpdateInfluence.clear();
	cout << "nodeNodeUpdates cleared count = " << nodeNodeCombUpdateInfluence.size() << "\n";
	// logLikelihood = 0;

	int previousParentNode, assignedParentNode;

	for(unsigned int i = 0; i < newSyntheticEvents.size(); i++)
	{
		int eventIndex = newSyntheticEvents[i][0];
		int eventNode = newSyntheticEvents[i][1];

		int eventParent = newSyntheticEvents[i][2];
		int eventTopic = newSyntheticEvents[i][3];

		int eventLevel = newSyntheticEvents[i][4];

		int assignedParent;

		if(eventLevel >= 0 &&  eventLevel <= maxTrainLevel)
		{
			
			double eventTime = eventIndexTimestamps[i];

			vector<ui> doc;
			// cout << "eventIndex -- " << eventIndex << endl;

			decreamentCountFromMatrices(eventIndex, eventNode, eventParent, eventTopic, doc, 0);
			assignedParent = getSampledParentAssignment(eventTime, eventNode, eventIndex, eventTopic, ITE);
			increamentCountToMatrices(eventIndex, eventNode, assignedParent, eventTopic, doc, 0);
			// assign the new sampled parent event to the event...
			
			newSyntheticEvents[eventIndex][2] = assignedParent;

		}
		else
		{
			assignedParent = invalidParId;
			newSyntheticEvents[eventIndex][2] = assignedParent;
			// updating for avgProbability of the parent events... at the end lets write it to a file...  
			if(ITE > BURN_IN && ITE%10 == 0)
			{
				for(ui i = 0; i < avgProbParForAllEvents[eventIndex].size(); i++)
				{
					avgProbParForAllEvents[eventIndex][i] += 1;
				}
			}
		}
		// if(i % 10000 == 0)
			// printf("Parent Assignment -- %d\n", i);

		if(ITE > BURN_IN && ITE%10 == 0)
		{
			// cout << "inside " << endl;
			parentDistEvery10thIter[eventIndex].push_back(assignedParent);
		}
	}

	// cout << "\nLog Likelihood = " << logLikelihood << "\n";

	return 0;
}

int getSampledParentAssignment(double eventTime, int eventNode, int eventIndex, int eventTopic, int ITE)
{
	// cout << "Getting sampled parent -- " << eventIndex << endl;
	int assignedParent = -2;

	// vector <ui> possibleParentEvents = getPossibleParentEvents(eventNode, eventIndex);
	vector <ui> possibleParentEvents = allPossibleParentEvents[eventIndex];
	vector<double> possibleParentExp = allPossibleParentEventsExponentials[eventIndex];
	// vector<double> possibleParentExp = computeParentExponentials(possibleParentEvents, eventIndex);
	
	// cout << "Got the candidate parents..." << endl;

	vector <double> calculatedProbVec;
	// cout << "Got the candidate parents...printing again.." << possibleParentEvents.size() << endl;

	// if(possibleParentEvents.size() == 0)
	// {
	// 	cout << " Hey no parents for you... \n";
	// }

	if(possibleParentEvents.size() > 0)
	{	
		// cout << "calling to calculate prob vevc...\n";
		calculatedProbVec = populateCalculatedProbVec(possibleParentEvents, possibleParentExp, eventNode, eventTopic, eventTime, ITE);
		// cout << "EventIndex -- " << eventIndex << "-- calculatedProbVec.size -- " << calculatedProbVec.size() << endl;

		// ui sampledIndex = getSampleFromMultinomial(calculatedProbVec);
		ui sampledIndex = getSampleFromDiscreteDist(calculatedProbVec);

		if(sampledIndex == calculatedProbVec.size()-1)
		{
			assignedParent = -1;
		}
		else
		{
			assignedParent = possibleParentEvents[sampledIndex];
		}

		logLikelihood += log(calculatedProbVec[sampledIndex]);
	}
	else
	{
		// cout << "inside else... as no poss parents...\n";
		assignedParent = -1;
		calculatedProbVec.push_back(1);
	}

	// updating for avgProbability of the parent events... at the end lets write it to a file...  
	if(ITE > BURN_IN && ITE%10 == 0)
	{
		// cout << "I have come here .. to write something..." << avgProbParForAllEvents[eventIndex].size() << " " << calculatedProbVec.size() << endl;
		for(ui i = 0; i < avgProbParForAllEvents[eventIndex].size(); i++)
		{
			avgProbParForAllEvents[eventIndex][i] += calculatedProbVec[i];
		}
	}

	if(assignedParent > (int)allEvents.size()-1)
	{
		cout << "some issue with parent sampling...\n";
		exit(0);
	}

	return assignedParent;
}

vector<double> populateCalculatedProbVec(vector <ui> possibleParentEvents, vector<double> possibleParentExp, int eventNode, int eventTopic, double eventTime, int ITE)
{
	// cout << "Populating prob vec...\n";
	vector <double> calculatedProbVec(possibleParentEvents.size() + 1, 0.0);
	
	// double firstTermNume, firstTermDenom, firstTerm, secondTerm;
	double firstTerm, secondTerm;

	double normalizationFactor = 0;

	unsigned int k;

	for(k = 0; k < possibleParentEvents.size(); k++)
	{
		vector <int> possParentEvent = newSyntheticEvents[possibleParentEvents[k]];
		int possParentNode = possParentEvent[1];
		int possParentEventTopic = possParentEvent[3];
		// cout << possParentNode << " " << possParentEventTopic << " " << endl;

		if (possParentEventTopic != invalidTopicId)
		{	
			firstTerm = getFirstTermOfParentAssignment(possParentEventTopic, eventTopic);

			double Wuv;
			Wuv = userUserInfluence[possParentNode][eventNode];

			if(Wuv > 0)
			{
				double temporalDecay = possibleParentExp[k];
				double infHazardTerm = Wuv * temporalDecay;
				double infSurvivalTerm = exp(-(Wuv));
				secondTerm = infSurvivalTerm * infHazardTerm;
			}
			else
			{
				secondTerm = 0;
			}

			calculatedProbVec[k] = firstTerm * secondTerm;
		}
		else
		{
			calculatedProbVec[k] = 0;	
		}
		normalizationFactor += calculatedProbVec[k];
	}

	// get prob of having no parent
	secondTerm = 0.0;
	firstTerm = getFirstTermOfParentAssignmentNoParent(eventNode, eventTopic);	

	double brHazardTerm = userBaseRateMap[eventNode];
	double brSurvivalTerm = exp(-(brHazardTerm * totalTime));

	secondTerm = brSurvivalTerm * brHazardTerm;

	calculatedProbVec[k] = firstTerm * secondTerm;
	normalizationFactor += calculatedProbVec[k];

	if(normalizationFactor == 0)
	{
		calculatedProbVec[calculatedProbVec.size() - 1] = 1;
		normalizationFactor = 1;
	}

	for(ui i = 0; i < calculatedProbVec.size(); i++)
	{
		calculatedProbVec[i] = calculatedProbVec[i] / normalizationFactor;
	}
	// cout << "Got calculated prob vec... \n";
	return calculatedProbVec;
}

double getFirstTermOfParentAssignment(int possParentEventTopic, int eventTopic)
{
	double firstTerm = 0.0;
	double firstTermNume = 0.0;
	double firstTermDenom = 1.0;
	
	firstTermNume = hyperBeta[eventTopic] + NTTCountVec[possParentEventTopic][eventTopic];
	
	firstTermDenom = sumBeta + NTTSumTopicsVec[possParentEventTopic];

	firstTerm = firstTermNume / firstTermDenom;

	return firstTerm;
}

double getFirstTermOfParentAssignmentNoParent(int eventNode, int eventTopic)
{
	double firstTerm = 0.0;
	double firstTermNume = 0.0;
	double firstTermDenom = 1.0;

	firstTermNume = hyperGamma[eventTopic] + NUTCountVec[eventNode][eventTopic];
	firstTermDenom = sumGamma + NUTSumTopicsVec[eventNode];
	firstTerm = firstTermNume / firstTermDenom;

	return firstTerm;
}

vector<ui> getPossibleParentEvents(int eventNode, int eventIndex)
{
	vector <ui> possibleParentEvents = allPossibleParentEvents[eventIndex];  
	return possibleParentEvents;
}


// Decreament Counters
int decreamentCountFromMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> doc, bool topicSampling)
{
	// cout << "decreament count from topic-topic matrix.. from the cell parentTopic -> eventTopic";
	// decreament only if the eventTopic is valid...
	if(eventTopic >= 0)
	{
		int eventParentTopic;

		if(eventParent >= 0)
		{
			eventParentTopic = newSyntheticEvents[eventParent][3];
			// printUnorderedMap("ttopic");
			decreamentCountFromTopicTopic(eventParentTopic, eventTopic);
			// printUnorderedMap("ttopic");
		}
		else if(eventParent == -1)
		{
			// printUnorderedMap("utopic");
			decreamentCountFromUserTopic(eventNode, eventTopic);
			// printUnorderedMap("utopic");
		}

		if(topicSampling == 1)
		{
			// decreament the count of words corresponding to this event...
			decreamentCountFromTopicWord(eventTopic, doc);
			
			// decreament counts corresponding to child events for topic sampling...
			decreamentCountsFromChildEvents(eventTopic, eventIndex);
		}
	}

	return 0;
}


int decreamentCountFromTopicTopic(int eventParentTopic, int eventTopic)
{
	// cout << "decreament count from topic-topic matrix.. from the cell parentTopic -> eventTopic";

	if(NTTCountVec[eventParentTopic][eventTopic] > 0)
	{
		NTTCountVec[eventParentTopic][eventTopic] -= 1;
	}

	if(NTTSumTopicsVec[eventParentTopic] > 0)
	{
		NTTSumTopicsVec[eventParentTopic] -= 1;
	}

	return 0;
}


int decreamentCountFromUserTopic(int eventNode, int eventTopic)
{
	// cout << "decreament count from user-topic matrix.. from the cell user -> eventTopic";

	if(NUTCountVec[eventNode][eventTopic] > 0)
	{
		NUTCountVec[eventNode][eventTopic] -= 1;	
	}

	if(NUTSumTopicsVec[eventNode] > 0)
	{
		NUTSumTopicsVec[eventNode] -= 1;	
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


int decreamentCountsFromChildEvents(int eventTopic, int eventIndex)
{
	// vector <int> childEventsList;
	vector <ui> childEventsList;
	try
	{
		childEventsList =  childEventsMap.at(eventIndex);
	}
	catch(exception &e)
	{
		return 0;
	}

	if(childEventsList.size() > 0)
	{
		for(unsigned int i = 0; i < childEventsList.size(); i++)
		{
			int childEventIndex = childEventsList[i];

			int childEventTopic = newSyntheticEvents[childEventIndex][3];

			if(childEventTopic != -1)
			{
				decreamentCountFromTopicTopic(eventTopic, childEventTopic);
			}
		}
	}

	return 0;
}


// Increament Counters
int increamentCountToMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector<ui> doc, bool topicSampling)
{
	// cout << "increament count to topic-topic matrix... in the cell parentTopic -> assignedTopic";

	if(eventParent >= 0)
	{
		int eventParentTopic = newSyntheticEvents[eventParent][3];
		// printUnorderedMap("ttopic");
		// cout << "increamenting...\n";
		increamentCountInTopicTopic(eventParentTopic, eventTopic);
		// printUnorderedMap("ttopic");
	}
	else if (eventParent == -1)
	{
		// printUnorderedMap("utopic");
		// cout << "increamenting...\n";
		increamentCountInUserTopic(eventNode, eventTopic);
		// printUnorderedMap("utopic");
	}

	if(topicSampling == 1)
	{
		// increament counts in topic-word...
		increamentCountInTopicWord(eventTopic, doc);
		// printUnorderedMap("psitopic");
		
		// increament counts corresponding to child events...
		increamentCountsForChildEvents(eventTopic, eventIndex);
	}

	return 0;
}

int increamentCountInTopicTopic(int eventParentTopic, int eventTopic)
{
	
	NTTSumTopicsVec[eventParentTopic] += 1;
	NTTCountVec[eventParentTopic][eventTopic] += 1;

	if(NTTCountVec[eventParentTopic][eventTopic] > newSyntheticEvents.size())
	{
		cout << "Somethings wrong.. the size of NTT is more than number of events...\n";
		cout << NTTCountVec[eventParentTopic][eventTopic] << " " << newSyntheticEvents.size() << "\n";
		exit(0);
	}


	if(NTTSumTopicsVec[eventParentTopic] > newSyntheticEvents.size())
	{
		cout << "Somethings wrong.. the size of NTT (sum) is more than number of events...\n";
		cout << NTTSumTopicsVec[eventParentTopic] << " " << newSyntheticEvents.size() << "\n";
		exit(0);
	}

	// NTSumTopics[eventParentTopic] += 1;

	// int cellIndex = getCellKey(eventParentTopic, eventTopic);
	// NTTCountMatrix[cellIndex] += 1;

	return 0;
}

int increamentCountInUserTopic(int eventNode, int eventTopic)
{
	NUTSumTopicsVec[eventNode] += 1;
	NUTCountVec[eventNode][eventTopic] += 1;
	
	if(NUTCountVec[eventNode][eventTopic] > newSyntheticEvents.size())
	{
		cout << "Somethings wrong.. the size of NUT is more than number of events...\n";
		cout << NUTCountVec[eventNode][eventTopic] << " " << newSyntheticEvents.size() << "\n";
		exit(0);
	}


	if(NUTSumTopicsVec[eventNode] > newSyntheticEvents.size())
	{
		cout << "Somethings wrong.. the size of NUT (sum) is more than number of events...\n";
		cout << NUTSumTopicsVec[eventNode] << " " << newSyntheticEvents.size() << "\n";
		exit(0);
	}


	// NUSumTopics[eventNode] += 1;

	// int cellIndex = getCellKey(eventNode, eventTopic);
	// NUTCountMatrix[cellIndex] += 1;

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

int increamentCountsForChildEvents(int eventTopic, int eventIndex)
{
	// vector <int> childEventsList;
	vector <ui> childEventsList;
	try
	{
		childEventsList = childEventsMap.at(eventIndex);
	}
	catch(exception &e)
	{
		return 0;
	}

	if(childEventsList.size() > 0)
	{
		// if(childEventsList.size() > 42)
			// printf("ChildEvents.size() -- %lu -- EventIndex -- %lu\n", childEventsList.size(), eventIndex);

		for(unsigned int i = 0; i < childEventsList.size(); i++)
		{
			int childEventIndex = childEventsList[i];

			int childEventTopic = newSyntheticEvents[childEventIndex][3];

			if(childEventTopic != -1)
			{
				increamentCountInTopicTopic(eventTopic, childEventTopic);
			}
		}
	}

	return 0;
}

///////////////////////////////
//////// UTIL FUNCTIONS ///////
///////////////////////////////

int getCellKey(int firstNum, int secondNum)
{
	// string keyString;

	// string firstUnderscore = to_string(firstPart) + "_";
	// keyString = firstUnderscore + to_string(secondPart);

	// using Szudziks Function....
	// a >= b ? a * a + a + b : a + b * b
	int keyVal = firstNum >= secondNum ? firstNum * firstNum + firstNum + secondNum : firstNum + secondNum * secondNum;



	// return keyString;
	return keyVal;
}

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


double getSampleFromGamma(double alpha, double beta)
{
	random_device rd;
	mt19937 gen(rd());
	gamma_distribution<double> distribution(alpha, beta);

	double gammaVal = distribution(gen);

	return gammaVal;
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


int writeEvery10ItersParentAssignmentToFile()
{
	ofstream parentAssignmentsFile;
	parentAssignmentsFile.open("parentAssignments.txt");
	// parentAssignmentsFile.open(configOutputFiles["parentAssignmentFile"]);

	cout << "Writing parent assignments to file ...\n";

	map<int, vector <int> >::iterator parentDistEvery10thIterIterator;

	for(parentDistEvery10thIterIterator = parentDistEvery10thIter.begin(); parentDistEvery10thIterIterator != parentDistEvery10thIter.end(); parentDistEvery10thIterIterator++)
	{
		int eventId = parentDistEvery10thIterIterator->first;
		vector<int> assignments = parentDistEvery10thIterIterator->second;

		parentAssignmentsFile << eventId << " ";

		for(ui i = 0; i < assignments.size(); i++)
		{
			parentAssignmentsFile << assignments[i] << " ";			
		}

		parentAssignmentsFile << endl;
	}

	return 0;
}

int writeAvgProbVectorsToFile()
{
	ofstream avgProbVecFile;
	avgProbVecFile.open("parentAssignmentAvgProbFile.txt");
	// avgProbVecFile.open(configOutputFiles["parentAssignmentAvgProbFile"]);

	cout << "Writing Avg Prob Vectors to file ...\n";
	// format
	// eid numofProbValues pid1 probval1 pid2 probval2 ...
	int takeAvgOver = (ITERATIONS - BURN_IN - 1)/10;
	for(ui i = 0; i < avgProbParForAllEvents.size(); i++)
	{
		avgProbVecFile << avgProbParForAllEvents[i].size() << " " << i << " ";

		ui j = 0;
		for(j = 0; j < avgProbParForAllEvents[i].size() - 1; j++)
		{
			avgProbVecFile << allPossibleParentEvents[i][j] << " " <<  avgProbParForAllEvents[i][j] / takeAvgOver << " ";
		}

		avgProbVecFile << i << " " << avgProbParForAllEvents[i][j] / takeAvgOver << "\n";
	}

	avgProbVecFile.close();
	return 0;
}


int writeAvgTopicProbVectorsToFile()
{
	ofstream avgTopicProbVecFile;
	// avgProbVecFile.open("avgParProb_OurModel.txt");
	avgTopicProbVecFile.open("B5_avgTopicProb_HMHP_estAll.txt");
	// avgTopicProbVecFile.open(configOutputFiles["topicAssignmentAvgProbFile"]);

	cout << "Writing Avg Prob Vectors to file ...\n";
	// format
	// eid numofProbValues pid1 probval1 pid2 probval2 ...
	for(ui i = 0; i < avgTopicProbVector.size(); i++)
	{
		avgTopicProbVecFile << avgTopicProbVector[i].size() << " " << i << " ";

		ui j = 0;
		for(j = 0; j < avgTopicProbVector[i].size(); j++)
		{
			ui topicId = j;
			avgTopicProbVecFile << topicId << " " <<  avgTopicProbVector[i][j] / 10 << " ";
		}

		avgTopicProbVecFile << "\n";
	}

	avgTopicProbVecFile.close();
	return 0;
}


int writeTopicTopicInteractionToFile()
{
	ofstream topicTopicIntOutFile;
	topicTopicIntOutFile.open("topicTopicInteraction.txt");
	// topicTopicIntOutFile.open(configOutputFiles["topicTopicInteractionFile"]);

	cout << "Writing topic-topic interactions ...\n";

	for(ui i = 0; i < NTTCountVec.size(); i++)
	{
		topicTopicIntOutFile << i << " ";

		for(ui j = 0; j < NTTCountVec[i].size(); j++)
		{
			topicTopicIntOutFile << NTTCountVec[i][j] << " ";
		}

		topicTopicIntOutFile << "\n";
	}

	topicTopicIntOutFile.close();
	
	return 0;
}

int writeUserUserInfluenceToFile()
{
	ofstream uuinfFile, uuinfFileAvg;
	uuinfFile.open("userUserInf.txt");
	uuinfFileAvg.open("userUserInfAvg.txt");
	// gUUInfFile.open(configOutputFiles["singleEdgeUserInfFile"]);

	cout << "Writing user user influences...\n";

	int uNode, vNode;

	for(ui i = 0; i < userGraphMap.size(); i++)
	{
		vector <ui> followers = userGraphMap[i];
		// cout << "At Node -- " << i << endl;
		uNode = i;

		if(followers.size() > 0)
		{
			uuinfFile << followers.size() << " " << uNode << " ";
			uuinfFileAvg << followers.size() << " " << uNode << " ";

			for(ui j = 0; j < followers.size(); j++)
			{
				vNode = followers[j];
				uuinfFile << vNode << " " << userUserInfluence[uNode][vNode] << " ";
				uuinfFileAvg << vNode << " " << userUserInfEvery10thIter[uNode][vNode] / avgOverNumChains << " ";
			}

			uuinfFile << "\n";
			uuinfFileAvg << "\n";
		}

	}

	uuinfFile.close();
	uuinfFileAvg.close();

	return 0;
}


int writeUserBaseRatesToFile()
{

	ofstream ubrFile;
	ubrFile.open("userBaseRate.txt");
	// ubrFile.open(configOutputFiles["userBaseRateFile"]);

	// map<ui, double>::iterator userBaseRateMapIt;

	// for(userBaseRateMapIt = userBaseRateMap.begin(); userBaseRateMapIt != userBaseRateMap.end(); userBaseRateMapIt++)
	// {
	// 	int uid = userBaseRateMapIt->first;
	// 	double ubrUid = userBaseRateMapIt->second;

	// 	// ubrFile << uid << " " << ubrUid << "\n";
	// 	ubrFile << setprecision(10) << uid << " " << ubrUid << "\n";
	// }


	for(int i = 0; i < maxNumNodes; i++)
	{
		int uid = i;
		double ubrUid = userBaseRateMap[i];

		if(ubrUid == 0)
		{
			ubrUid = defaultUserBaseRate;	
		}

		// ubrFile << uid << " " << ubrUid << "\n";
		ubrFile << setprecision(10) << uid << " " << ubrUid << "\n";
	}

	ubrFile.close();

	return 0;
}


int writeNodeNodeCountAndNodeCount()
{
	ofstream wuvAnalyze;
	wuvAnalyze.open("B5_Nuv_Nu_Wuv_HMHP_estAll.txt");

	unordered_map<ui, unordered_map<ui, ui> >::iterator nodeNodeCountIt;

	for (nodeNodeCountIt = nodeNodeCount.begin(); nodeNodeCountIt != nodeNodeCount.end(); nodeNodeCountIt++)
	{
		ui uNode = nodeNodeCountIt->first;

		unordered_map<ui, ui> edgeCountOverNode = nodeNodeCountIt->second;

		unordered_map<ui, ui>::iterator edgeCountOverNodeIt;

		for(edgeCountOverNodeIt = edgeCountOverNode.begin(); edgeCountOverNodeIt != edgeCountOverNode.end(); edgeCountOverNodeIt++)
		{
			ui vNode = edgeCountOverNodeIt->first;
			ui Nuv = edgeCountOverNodeIt->second;

			ui Nu = nodeEventsCountMap[uNode];

			double wuv = ((Nuv + 0.01) * 1.0)/(Nu + 1);

			wuvAnalyze << vNode << " " << uNode << " " << Nuv << " " << Nu << " " << wuv << "\n";
		}
	}

	wuvAnalyze.close();

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

// unordered_map <ll, vector <ll> > readIntVectorMapFromFile(string fileName)
vector < vector <ui> > readIntVectorMapFromFile(string fileName)
{
	ifstream intVectorFile;
	intVectorFile.open(fileName);
	// intVectorFile.open("mapped_users_followers_graph_sorted_1000000_users.txt");

	// unordered_map <ll, vector <ll> > intVectorMap;
	vector < vector <ui> > intVectorMap(maxNumNodes);

	string line;
	stringstream ss;
	ui nodeId;
	ui follower;
	ui count; 
	ui mapsize = 0;

	if (intVectorFile.is_open())
	{
		while(getline(intVectorFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;
			ss >> count >> nodeId;

			vector <ui> tempVec;
			for(unsigned i = 0; i < count; i++)
			{
				ss >> follower;
				tempVec.push_back(follower);
			}
			sort(tempVec.begin(), tempVec.end());

			intVectorMap[nodeId] = tempVec;
			// intVectorMap.push_back(tempVec);

			if(mapsize % 500000 == 0)
			{
				cout << "Reading File... Done with -- " << fileName << " -- " << mapsize << endl;
			}

			mapsize++;
		}

		intVectorFile.close();
		cout << "read -- " << fileName << endl;
	}
	else
	{
		cout << "Error opening file -- " << fileName << "\n";
	}
	return intVectorMap;
}

vector< vector <double> > readMultipleDoubleVectorsFromFile(string fileName)
{
	vector< vector <double> > vecVecdouble;
	ifstream vecVecdoubleFile;
	vecVecdoubleFile.open(fileName);

	string line;
	stringstream ss; 

	if(vecVecdoubleFile.is_open())
	{
		while(getline(vecVecdoubleFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;
			vector<double> tempVec;
			double tempVal;

			while(ss >> tempVal)
			{
				tempVec.push_back(tempVal);
			}

			vecVecdouble.push_back(tempVec);
		}

		vecVecdoubleFile.close();
	}
	else
	{
		cout << "Error opening file -- " << fileName << "\n";
	}
	return vecVecdouble;
}

vector <double>  readDoubleVectorFromFile(string fileName)
{
	vector <double> doubleVec;
	ifstream doubleVecFile;
	doubleVecFile.open(fileName);

	string line; 
   	stringstream ss; 

	ss.clear();
	ss.str("");

	if(doubleVecFile.is_open())
	{
		getline(doubleVecFile, line);

		ss << line;
		double tempVal;
		while(ss >> tempVal)
		{
			doubleVec.push_back(tempVal);
		}
	}
	else
	{
		cout << "Error opening file -- " << fileName << "\n";
	}

	return doubleVec;
}


map<ui, double> getUserBaseRates(string fileName)
{
	map<ui, double> ubRateMap;

	ifstream ubrFile;
	ubrFile.open(fileName);

	string line;
	stringstream ss;

	ui uid, twCount;			// time1, time2;
	double ubr;

	while(getline(ubrFile, line))
	{
		ss.clear();
		ss.str("");

		ss << line;

		ss >> uid >> twCount >> ubr;

		ubRateMap[uid] = ubr;
	}

	ubrFile.close();

	return ubRateMap;
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


int populateParentEventsForAllFromFile(string parentsFile, string parentExpFile)
{
	ifstream possParFile;
	possParFile.open(parentsFile);

	string line;
	stringstream ss;

	ui count, eid;
	if(possParFile.is_open())
	{

		while(getline(possParFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;

			ss >> count >> eid;

			vector<ui> parEvents;
			ui tempPar;
			for(ui i = 0; i < count; i++)
			{
				ss >> tempPar;
				parEvents.push_back(tempPar);
			}

			allPossibleParentEvents.push_back(parEvents);
			parEvents.clear();
		}

		possParFile.close();
	}
	else
	{
		cout << "Cannot open the possible parent events file\n";
		exit(0);
	}

	

	ifstream possExpFile;
	possExpFile.open(parentExpFile);

	if(possExpFile.is_open())
	{
		while(getline(possExpFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;

			ss >> count >> eid;
			vector <double> expEvents;
			double tempExp;
			for(ui i = 0; i < count; i++)
			{
				ss >> tempExp;
				expEvents.push_back(tempExp);
			}

			allPossibleParentEventsExponentials.push_back(expEvents);
			expEvents.clear();
		}

		possExpFile.close();
	}
	else
	{
		cout << "Cannot open the possible parent events Exponentials file\n";
		exit(0);
	}
	
	return 0;
}


unordered_map <string, string> getConfigInputOutputFileNames(string fileName)
{
	ifstream inputOutputFiles;
	inputOutputFiles.open(fileName);

	unordered_map <string, string> localFilePathsObj;

	if(inputOutputFiles.is_open())
	{	
		string line, fileKey, fname;
		stringstream ss;

		while(getline(inputOutputFiles, line))
		{
			ss.clear();
			ss.str("");

			ss << line;
			ss >> fileKey >> fname;

			localFilePathsObj[fileKey] = fname;
		}
	}
	else
	{
		cout << "Issue opening the file containing the path to input files... Exiting here...\n";
		exit(0);
	}

	return localFilePathsObj;
}
