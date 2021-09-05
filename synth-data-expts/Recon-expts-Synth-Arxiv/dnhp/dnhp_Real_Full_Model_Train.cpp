#include <iostream>				//for basic C++ functions 
#include <cstdio>				//for std C functions
#include <fstream>				//for i/o stream
#include <sstream>				//for parse data using stringstreams...
#include <string>				//for C++ string functions
#include <cstring>				//we need this for memset... and string functions from C
#include <unordered_map>		//to maintain the map of user-tweet_count, and user_user_tweet_count
#include <map>					//to maintain the map of user-tweet_count, and user_user_tweet_count
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
int numTopics = 100;
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
double baseAlphaTopic = 1; 

double logLikelihood = 0;


// Read data Functions...
vector < vector <int> > getEventsFromFile(string fileName);
vector < vector <ui> > getSyntheticDocsFromFile(string fileName);
vector < vector <ui> > readIntVectorMapFromFile(string fileName);
unordered_map <int, unordered_map<int, float> > populateThetaPriorForTopicPairs(string fileName);
unordered_map <ui, int> getDegreeVals(unordered_map <ui, unordered_map<ui, int> > temp2LevelMap);
int populateParentEventsForAllFromFile(string parentsFile, string parentExpFile);
int getEntityPerTweet(string filename);

// Initialization Functions...
// vector< vector <int> > initializeForSampler(vector< vector <int> > allEvents, int flag);
vector< vector <int> > initializeEvents(vector< vector <int> > allEvents, int flag);
int initializeTopics();
int initializeTopicTopicCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeTopicMentionCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeTopicWordCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeUserUserInfluence();
int populateListOfEmptyEvents();

int initializeAllCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeBaseRateAndInfluence();
int initializeBaseRates();
int initializeNNTTCounts(vector< vector <int> > localNewSyntheticEvents);
int initializeUserUserInfluence();
int initializeTopicTopicInfluence();
int initializeUserTopicInfluence();
int initializeAvgProbabilityVectors();

// Topic Assignment related functions...
int sampleTopicAssignment(int ITE);
int getSampledTopicAssignment(int eventIndex, int eventNode, int eventParent, vector<ui> currDoc, vector<ui> currMention, int ITE);

double getEventGenerationTerm(int eventParentIndex, int eventParentNode, int eventParentTopic, int eventIndex, int eventNode, int topicId);
double getBaseRateTerm(int topicId, int eventNode);
double getChildEventsTerm(int eventIndex, int topicId, int eventNode);

// vector <int> getPossibleSetOfTopics(int eventIndex, int eventParent, vector <ui> currDoc, vector <ui> currMention);
// vector <int> getCombinedSetOfTopics(vector <int> topicsFromParent, vector <int> topicsFromWords, vector <int>topicsFromMentions, vector <int> topicsFromChildEvents);
// vector <int> getSetOfTopicsFromParentTopic(int eventParent);
// vector <int> getSetOfTopicsFromChildEvents(int eventIndex);
// vector <int> getSetOfTopicsFromMentions(vector <ui> currMention);
// vector <int> getSetOfTopicsFromWords(vector <ui> currDoc);
// unordered_map <int, ui> getHistOfTopicsOverChildEvents(int eventIndex);

// int updateReverseTopicMapsAndChildEventsMap();

// double getChildTopicTopicTerm(int eventIndex, int topicId);
// double getFLChildEventTerm(int topicId, int childEventsWNeighTopics, int childEventsWNonNeighTopics);
// double getSLChildEventTerm(int topicId, unordered_map <int, ui> childEventTopicsHist);
// double computeEachTerm(double betaValue, int numChildTransitions, int totalTransitions);

// double getTopicTopicTerm(int topicId, int eventParentTopic, int eventParent);
// vector<double> getFLTopicTopicTerm(int eventParentTopic);
// double getNeighborTopicTopicFLTerm(int eventParentTopic);
// double getNonNeighborTopicTopicFLTerm(int eventParentTopic);

// double getTopicMentionTerm(int topicId, int eventIndex, vector<ui> currMention);

double getTopicWordTerm(int topicId, int eventIndex, vector<ui> currDoc);

// double getUserTopicTerm(int topicId, int eventNode);

int sampleUserUserInfluence(int ITE);
int sampleTopicTopicInfluence(int ITE);
int sampleUserTopicInfluence(int ITE);
int updateNNTTCountMap();

int updateUserBaseRates();
int updateTopicBaseRates();

int sampleParentAssignment(int ITE);
int getSampledParentAssignment(double eventTime, li eventNode, li eventIndex, li eventTopic, int ITE);
vector<double> populateCalculatedProbVec(vector <ui> possibleParentEvents, vector<double> possibleParentExp, int eventNode, int eventTopic, double eventTime, int ITE);
double getFirstTermOfParentAssignment(int possParentEventTopic, int eventTopic);
double getFirstTermOfParentAssignmentNoParent(int eventNode, int eventTopic);

int createWordHistForAllDocs();
int createMentionHistForAllDocs();
unordered_map <ui, ui> getHistOfWordsOverWordsFromDoc(vector<ui> currDoc);


// Decreamenting counts from matrices...
int decreamentCountFromMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> currDoc, vector<ui> currMention, bool topicSampling);
int decreamentCountFromTopicTopic(int eventParentTopic, int eventTopic);
int decreamentCountFromTopicWord(int eventTopic, vector <ui> doc);
int decreamentCountFromTopicMentions(int eventTopic, vector<ui> currMention);
int decreamentCountFromUserTopic(ui eventNode, int eventTopic);
int decreamentCountsFromChildEvents(int eventTopic, int eventIndex);


// Increamenting counts to the matrices...
int increamentCountToMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> currDoc, vector<ui> currMention, bool topicSampling);
int increamentCountInTopicTopic(int eventParentTopic, int eventTopic);
int increamentCountInTopicMentions(int eventTopic, vector<ui> currMention);
int increamentCountInTopicWord(int eventTopic, vector<ui> currDoc);
int increamentCountInUserTopic(ui eventNode, int eventTopic);
int increamentCountsForChildEvents(int eventTopic, int eventIndex);


int getSampleFromMultinomial(vector<double> calculatedProbVec);
vector<double> getNormalizedLogProb(vector<double> calculatedProbVec);
int getSampleFromDiscreteDist(vector<double> normalizedProbVector);
double getSampleFromGamma(double alpha, double beta);


// Writing outputs...
map <int, vector<int> > topicDistEvery10thIter, parentDistEvery10thIter;
vector <vector <double> > avgProbParForAllEvents;
int writeEvery10ItersTopicAssignmentToFile();
int writeEvery10ItersParentAssignmentToFile();
int writeAvgProbVectorsToFile();
int writeUserUserInfluenceToFile();
int writeTopicTopicInfluenceToFile();
int writeUserTopicInfluenceToFile();
int writeUserBaseRatesToFile();
int writeTopicBaseRatesToFile();


// data structures for inputs...
// train data...
vector< vector <int> > allEvents;
vector< vector <ui> > allEventsDocs;
vector< vector <ui> > allEventsMentions;

vector< vector <int> > newSyntheticEvents;
vector< vector <int> > allEntitiesPerTweet;
vector<int> entityPerTweet;


// Event timestamps
unordered_map<int, li> eventIndexTimestamps;

// Possible parents and exponential rates (probability)...
vector <vector <ui> > allPossibleParentEvents;
vector <vector <double> > allPossibleParentEventsExponentials;


// Count Matrices...
// Dir-Tree priors
// At the first level count -- 0 = neighbors, 1 = non-neighbors
unordered_map <int, unordered_map<int, int> > NTTopicsSecLevelCount;
unordered_map <int, int > NTTopicsSumTopicsVec;
unordered_map <int, unordered_map<int, int> > NTMentionsSecLevelCount;
unordered_map <int, int > NTMentionsSumMentionsVec;

// Dir distributions prior
unordered_map <int, unordered_map<ui, int> > NTWCountVec;
unordered_map<int, int> NTWSumWordsVec;
unordered_map <ui, unordered_map<ui, int> > NUTCountVec;
unordered_map<ui, int> NUTSumTopicsVec;

unordered_map <ui, unordered_map<ui, int> > unifiedNodeTopicCount;

unordered_map <ui, unordered_map<ui, int> > mentionTopicsMap;
unordered_map <ui, unordered_map<ui, int> > reverseTopicTopicTransitions;
unordered_map <ui, unordered_map<ui, int> > reverseWordTopicsMapping;
unordered_map <ui, unordered_map<ui, int> > reverseMentionTopicsMapping;

vector < vector <ui> > wordHistAllDocsVector;
vector < vector <ui> > mentionHistAllDocsVector;
map <ui, vector<ui> > childEventsMap;

unordered_map<ui, unordered_map<ui, ui> > nodeNodeCount;
unordered_map<ui, ui>  nodeNodeCountSum;

unordered_map<ui, unordered_map<ui, ui> > topicTopicCount;
unordered_map<ui, ui>  topicTopicCountSum;

unordered_map<ui, unordered_map<ui, ui> > nodeTopicCountMap;
unordered_map<ui, unordered_map<ui, ui> > topicNodeCountMap;
unordered_map<ui, unordered_map<ui, ui> > nodeTopicCountFromWhere;
unordered_map<ui, unordered_map<ui, ui> > nodeTopicCountToWhere;

map<ui, vector<ui> > nodeEventsMap;
vector<ui> nodeEventsCountMap(maxNumNodes, 0);
vector<ui> topicEventsCountMap(maxNumTopics, 0);

unordered_map <ui, unordered_map<ui, double> > userUserInfluence;
unordered_map <ui, unordered_map<ui, double> > userUserInfEvery10thIter;
unordered_map <ui, double> userUserInfluenceSumPerNode;

unordered_map <ui, unordered_map<ui, double> > topicTopicInfluence;
unordered_map <ui, unordered_map<ui, double> > topicTopicInfEvery10thIter;
unordered_map <ui, double> topicTopicInfluenceSumPerTopic;

unordered_map <ui, unordered_map<ui, double> > userTopicInfluence;
unordered_map <ui, unordered_map<ui, double> > userTopicInfEvery10thIter;
unordered_map <ui, double> userTopicInfluenceSumPerNode;

unordered_map <ui, unordered_map<ui, double> > NuNodekTopicWuNodevNodeSumUNode;
unordered_map <ui, unordered_map<ui, double> > NuNodekTopicTkTopickPrimeTopicUKPrime;


vector < vector <ui> > userGraphMap;
vector < vector <ui> > reverseUserGraphMap;
// unordered_map <ui, unordered_map<ui, int> > userGraphMap;
// unordered_map <ui, unordered_map<ui, int> > reverseUserGraphMap;

unordered_map <int, unordered_map<int, float> > topicTopicThetaParamPrior;

unordered_map<int, int> emptyEvents;

// double muVal = 0.00002;
double defaultMuVal;
double defaultUserBaseRate, defaultTopicBaseRate;

map<ui, double> userBaseRateMap, topicBaseRateMap;

double possTopicTime = 0;
double totalTime;

int maxTrainLevel = 3;
int percentLessCount = 70;

ofstream wuvFile, tkkFile;

double eGenTermTime, twTermTime, chGenTermTime;

int avgOverNumChains = 0;

int invalidTopicId = 1000;
int invalidParId = -2;


int main()
{
	// wuvFile.open("wuvOutFile_estAll_noLevel.txt");
	// tkkFile.open("tkkOutFile_estAll_noLevel.txt");

	ofstream llFile;
	llFile.open("estAllLogLikelihood.txt");

	// Read all the events -- "tStamp uid -1 -1"
	allEvents = getEventsFromFile("../../centralFiles/eventsFile_scaledTime_250K.txt");
	// allEvents = getEventsFromFile("./centralFiles/events_10.txt");
	cout << "Got all the events --- " << allEvents.size() << endl;
	
	// Read all the docs...
	allEventsDocs = getSyntheticDocsFromFile("../../centralFiles/dataFile_250K.txt");
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
	cout << "Populate list of empty events..." << endl;
	populateListOfEmptyEvents();
	cout << "Got the list of empty events..." << emptyEvents.size() << endl;


	// cout << "Get the prior theta parameter for topic pairs....";
	// topicTopicThetaParamPrior = populateThetaPriorForTopicPairs("priorTopicTopicInfluenceFile.txt");
	// cout << "Got the prior theta parameter for the topic pairs" << topicTopicThetaParamPrior.size() << "\n";

	// get mention-topic reverse map...
	// cout << "Getting Mention-Topic Reverse Map..." << endl;
	// mentionTopicsMap = readIntVectorMapFromFile("reverseEntityMentionsMapping_Ided.txt");
	// cout << "Mapsize = " << mentionTopicsMap.size() << endl;
	
	// Initialize the events...
	int flag = 1;
	// newSyntheticEvents = initializeForSampler(allEvents, flag);
	newSyntheticEvents = initializeEvents(allEvents, flag);
	initializeTopics();
	initializeAllCounts(newSyntheticEvents);
	initializeBaseRateAndInfluence();
	initializeAvgProbabilityVectors();

	for(int ITE = 0; ITE < ITERATIONS; ITE++)
	{
		logLikelihood = 0;

		high_resolution_clock::time_point t1, t2;
		auto duration = 0;
 		
		eGenTermTime = 0;
		twTermTime = 0;
		chGenTermTime = 0;

		t1 = high_resolution_clock::now();
		sampleTopicAssignment(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();

		cout << "sampled topics for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;
		cout << "Detailed Time: " << eGenTermTime << " " << twTermTime <<  " " << chGenTermTime << " " << duration << endl;

		// cout << "poss Topic time = " << possTopicTime << endl;

		t1 = high_resolution_clock::now();
		sampleParentAssignment(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "sampled parents for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;
		
		t1 = high_resolution_clock::now();
		updateNNTTCountMap();
		t2 = high_resolution_clock::now();

		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "Updating Child Events Map and Node-Node Count after -- "<< ITE << " Iteration --  Time  taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		sampleUserUserInfluence(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "Sampled User-User Influence for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		sampleTopicTopicInfluence(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "Sampled Topic-Topic Influence for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		sampleUserTopicInfluence(ITE);
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "Sampled User-Topic Influence for iteration -- "<< ITE << "--  Time taken -- " << duration << endl;

		t1 = high_resolution_clock::now();
		updateUserBaseRates();
		t2 = high_resolution_clock::now();

		t1 = high_resolution_clock::now();
		updateTopicBaseRates();
		t2 = high_resolution_clock::now();
	
		duration = duration_cast<microseconds>( t2 - t1 ).count();
		cout << "Updating Alpha Value and User Base Rates -- "<< ITE << "--  Time taken -- " << duration << endl;
		// cout << "Default Mu Val = " << defaultMuVal << endl;
		cout << "Default Mu Val = " << defaultUserBaseRate << " " << defaultTopicBaseRate << endl;

		cout << setprecision(10) << "Log Likelihood = " << logLikelihood << "\n";
		llFile << setprecision(10) << logLikelihood << "\n";
	}
	
	llFile.close();

	writeEvery10ItersTopicAssignmentToFile();
	writeEvery10ItersParentAssignmentToFile();
	writeAvgProbVectorsToFile();
	writeUserUserInfluenceToFile();
	writeTopicTopicInfluenceToFile();
	writeUserTopicInfluenceToFile();
	writeUserBaseRatesToFile();
	writeTopicBaseRatesToFile();

	cout << "Avg over num chains : " << avgOverNumChains << endl;

	return 0;
}

// Initialization... Assign parents and topics... Also the User-User Influence...
// vector< vector <int> > initializeForSampler(vector< vector <int> > allEvents, int flag)
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

				if(allEvents[candParId][4] >= 0 && allEvents[candParId][4] <= maxTrainLevel)
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
		
		// if(eLevel >= 0 && eLevel <= maxTrainLevel)
		if(eLevel <= maxTrainLevel)
		{
			currDoc = allEventsDocs[eventIndex];

			vector<double> topicAssignmentScores;

			for(int topicId = 0; topicId < numTopics; topicId++)
			{
				double topicWordTerm = getTopicWordTerm(topicId, eventIndex, currDoc);
				topicAssignmentScores.push_back(topicWordTerm);
			}

			int assignedTopicInd =  getSampleFromMultinomial(topicAssignmentScores);
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

	int countNuFor = (int)(percentLessCount * localNewSyntheticEvents.size() * 0.01);
	cout << "Count Nu For --- initialize --- " << countNuFor << "\n";
	// unordered_map<ui, unordered_map<ui, ui> > tempMap;

	for(ui i = 0; i < localNewSyntheticEvents.size(); i++)
	{
		int eNode = localNewSyntheticEvents[i][1];
		int eParent = localNewSyntheticEvents[i][2];
		int eTopic = localNewSyntheticEvents[i][3];
		int eLevel = localNewSyntheticEvents[i][4];
        // cout << "event id == " << i << endl;
		// if(eLevel >= 0 && eLevel < maxTrainLevel)
		if(i < countNuFor)
		{
			nodeTopicCountMap[eNode][eTopic] += 1;
			topicNodeCountMap[eTopic][eNode] += 1;

			nodeEventsCountMap[eNode] += 1;
			topicEventsCountMap[eTopic] += 1;

			// tempMap[eNode][eTopic] = tempMap[eNode][eTopic] + 1;
		}
		
		// if(eLevel >= 0 && eLevel <= maxTrainLevel)
		// {
        if(eParent > -1)
        {
            int eParentNode = localNewSyntheticEvents[eParent][1];
            int eParentTopic = localNewSyntheticEvents[eParent][3];

            // maintaining the reverse topic-topic transition map...
            // reverseTopicTopicTransitions[eTopic][eParentTopic] = 1;
            
            // Topic Topic counts...			
            topicTopicCount[eParentTopic][eTopic] += 1;
            
            // Sum at second level -- Do we really need the sum...?
            topicTopicCountSum[eParentTopic] += 1;

            // Node Node counts...
            nodeNodeCount[eParentNode][eNode] += 1;

            nodeTopicCountFromWhere[eNode][eParentTopic] += 1;
            nodeTopicCountToWhere[eNode][eTopic] += 1;

            // Node Sum counts --  Do we really need the sum...?
            nodeNodeCountSum[eParentNode] += 1;

            childEventsMap[eParent].push_back(i);

            sumAll++;
        }
        else
        {
            NUTCountVec[eNode][eTopic] += 1;
            NUTSumTopicsVec[eNode] += 1;
        }
		// }

	}

	cout << "Number of Edges = " << sumAll << "\n";
	// cout << "Number of non-zero topic-topic maps -- " << NTTopicsSecLevelCount.size() << endl;
	cout << "Number of non-zero topic-topic maps -- " << topicTopicCount.size() << endl;

	return 0;
}


int initializeBaseRateAndInfluence()
{
	initializeUserUserInfluence();
	initializeTopicTopicInfluence();
	initializeUserTopicInfluence();

	initializeBaseRates();
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

			// cout << "At child -- " << j << endl;

			double shapeParam = nodeNodeCount[uNode][vNode] + baseAlpha;

			// double infVal  = shapeParam * scaleParam;
			double infVal = getSampleFromGamma(shapeParam, scaleParam);

			tempNodeInf[vNode] = infVal;

			userUserInfluenceSumPerNode[uNode] += infVal;
			avgVal += infVal;
		}

		userUserInfluence[uNode] = tempNodeInf;
		tempNodeInf.clear();

		count++;

		if(count % 100000 == 0)
			cout << "Initialized for " <<  count << " users...\n";
	}

	cout << "Avg Wuv val = " << avgVal / count << endl; 

	cout << "Initialized user user influence\n";
	cout << "Added Nu number of times = " << numTimesAddedNu << endl;
	return 0;
}

int initializeTopicTopicInfluence()
{
	cout << "Initializing topic topic influence\n";

	int count = 0;
	int kTopic, kPrimeTopic;

	double avgVal = 0.0;

	for(int i = 0; i < numTopics; i++)
	{
		kTopic = i;

		// double scaleParam = 1/(topicEventsCountMap[kTopic] + baseBeta);
		
		unordered_map <ui, double> tempNodeInf;

		for(int j = 0; j < numTopics; j++)
		{
			kPrimeTopic = j;

			// float betaVal = 1.0/topicTopicThetaParamPrior[kTopic][kPrimeTopic];
			float betaVal = 1.0/2;

			double scaleParam = 1/(topicEventsCountMap[kTopic] + betaVal);
			double shapeParam = topicTopicCount[kTopic][kPrimeTopic] + baseAlphaTopic;
				
			double infVal = getSampleFromGamma(shapeParam, scaleParam);

			tempNodeInf[kPrimeTopic] = infVal;

			topicTopicInfluenceSumPerTopic[kTopic] += infVal;

			avgVal += infVal;
		}

		topicTopicInfluence[kTopic] = tempNodeInf;
		tempNodeInf.clear();

		count++;
	}

	cout << "Avg Tkk val = " << avgVal / count << endl;
	cout << "Initialized Topic Topic influence...\n";

	return 0;
}



int initializeUserTopicInfluence()
{
	cout << "Initializing user topic influence\n";

	int count = 0;
	int uNode, kTopic;

	double avgVal = 0.0;

	for(int i = 0; i < numNodes; i++)
	{
		uNode = i;
		// double scaleParam = 1/(topicEventsCountMap[kTopic] + baseBeta);
		
		unordered_map <ui, double> tempNodeInf;

		for(int j = 0; j < numTopics; j++)
		{
			kTopic = j;

			// float betaVal = 1.0/topicTopicThetaParamPrior[kTopic][kPrimeTopic];
			float betaVal = 1.0/2;

			double scaleParam = 1/(nodeEventsCountMap[uNode] + betaVal);
            // double shapeParam = nodeTopicCountMap[uNode][kTopic] + baseAlphaTopic;
            double shapeParam = nodeTopicCountToWhere[uNode][kTopic] + baseAlphaTopic;

			double infVal = getSampleFromGamma(shapeParam, scaleParam);

			tempNodeInf[uNode] = infVal;

			userTopicInfluenceSumPerNode[uNode] += infVal;

			avgVal += infVal;
		}

		userTopicInfluence[uNode] = tempNodeInf;
		tempNodeInf.clear();

		count++;
	}

	cout << "Avg Uuk val = " << avgVal / count << endl;
	cout << "Initialized User Topic influence...\n";

	return 0;
}

int updateNNTTCountMap()
{
	int eventNode, eventTopic, eventLevel, parentEvent, parentNode, parentTopic;
	
	nodeNodeCount.clear();
	nodeNodeCountSum.clear();

	topicTopicCount.clear();
	topicTopicCountSum.clear();

	nodeTopicCountMap.clear();
	topicNodeCountMap.clear();

	nodeTopicCountFromWhere.clear();
	nodeTopicCountToWhere.clear();

	childEventsMap.clear();

	int countNuFor = (int)(percentLessCount * newSyntheticEvents.size() * 0.01);
	cout << "Count Nu For --- " << countNuFor << "\n";

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{
		eventNode = newSyntheticEvents[i][1];
		parentEvent = newSyntheticEvents[i][2];
		eventTopic = newSyntheticEvents[i][3];
		eventLevel = newSyntheticEvents[i][4];

		if (i < countNuFor)
		// if(eventLevel >= 0 && eventLevel < maxTrainLevel)
		{
			nodeTopicCountMap[eventNode][eventTopic] += 1;
			topicNodeCountMap[eventTopic][eventNode] += 1;
		}

		// if(eventLevel >= 0 && eventLevel <= maxTrainLevel)
		// {
        if(parentEvent >= 0)
        {
            parentNode = newSyntheticEvents[parentEvent][1];
            parentTopic = newSyntheticEvents[parentEvent][3];

            nodeNodeCount[parentNode][eventNode]++;
            nodeNodeCountSum[parentNode] += 1;

            topicTopicCount[parentTopic][eventTopic]++;
            topicTopicCountSum[parentTopic] += 1;

            nodeTopicCountFromWhere[eventNode][parentTopic] += 1;
            nodeTopicCountToWhere[eventNode][eventTopic] += 1;

            childEventsMap[parentEvent].push_back(i);
        }
		// }
	}

	return 0;
}

int initializeBaseRates()
{
	int totalSponCount = 0;
	int minTime = eventIndexTimestamps[0];
	int maxTime = eventIndexTimestamps[eventIndexTimestamps.size() - 1];

	int totalDataObservedTime = (maxTime - minTime) / timeScalingFactor;
	// int totalDataObservedTime = (maxTime - minTime);

	map<int, int> eachNodeSponCount;
	map<int, int> eachTopicSponCount;
	map<int, bool> distinctNodes;
	map<int, bool> distinctTopics;

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{
		// if(newSyntheticEvents[i][4] >= 0 && newSyntheticEvents[i][4] <= maxTrainLevel)
		if(newSyntheticEvents[i][4] <= maxTrainLevel)
		{
			if(newSyntheticEvents[i][2] == -1)
			{
				totalSponCount++;
				distinctNodes[newSyntheticEvents[i][1]] = 1;
				distinctTopics[newSyntheticEvents[i][3]] = 1;
				eachNodeSponCount[newSyntheticEvents[i][1]]++;
				eachTopicSponCount[newSyntheticEvents[i][3]]++;
			}
		}
	}

	double averageUBR = 0, averageTBR = 0;

	double minMuVal = 1000000;
	double maxMuVal = 0;

	defaultUserBaseRate = (totalSponCount * 1.0) / (totalDataObservedTime * distinctNodes.size());

	for(int nodeId = 0; nodeId < maxNumNodes; nodeId++)
	{
		int nodeSponCount = eachNodeSponCount[nodeId];
		if(nodeSponCount > 0)
		{
			userBaseRateMap[nodeId] = (nodeSponCount * 1.0) / totalDataObservedTime;
		}
		else
		{
			userBaseRateMap[nodeId] = defaultUserBaseRate;
		}

		if(userBaseRateMap[nodeId] > maxMuVal)
		{
			maxMuVal = userBaseRateMap[nodeId];
		}

		if(userBaseRateMap[nodeId] < minMuVal)
		{
			minMuVal = userBaseRateMap[nodeId];
		}

		averageUBR +=  userBaseRateMap[nodeId];
	}

	averageUBR = averageUBR / userBaseRateMap.size();
	
	cout << "totalSponCount = " << totalSponCount << " minMuVal = " << minMuVal << " maxMuVal = " << maxMuVal << " MLE MuVal = " << defaultUserBaseRate << endl;

	defaultTopicBaseRate = (totalSponCount * 1.0) / (totalDataObservedTime * distinctNodes.size());

	for(int topicId = 0; topicId < numTopics; topicId++)
	{
		int topicSponCount = eachTopicSponCount[topicId];
		if(topicSponCount > 0)
		{
			topicBaseRateMap[topicId] = (topicSponCount * 1.0) / totalDataObservedTime;
		}
		else
		{
			topicBaseRateMap[topicId] = defaultTopicBaseRate;
		}

		if(topicBaseRateMap[topicId] > maxMuVal)
		{
			maxMuVal = topicBaseRateMap[topicId];
		}

		if(topicBaseRateMap[topicId] < minMuVal)
		{
			minMuVal = topicBaseRateMap[topicId];
		}

		averageUBR +=  topicBaseRateMap[topicId];
	}

	averageTBR = averageTBR / topicBaseRateMap.size();
	
	cout << "totalSponCount = " << totalSponCount << " minMuVal = " << minMuVal << " maxMuVal = " << maxMuVal << " MLE MuVal = " << defaultTopicBaseRate << endl;

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


///////////////////////////////////////////////////
/////// SAMPLE USER USER INFLUENCE/////////////////
///////////////////////////////////////////////////

int sampleUserUserInfluence(int ITE)
{
	unordered_map<ui, unordered_map<ui, ui> >::iterator nodeNodeCountIterator;
	double avgVal = 0;
	int count = 0;
	int uNode, vNode;

	userUserInfluenceSumPerNode.clear();

    for(int vNode = 0; vNode < maxNumNodes; vNode++)
    {
        NuNodekTopicWuNodevNodeSumUNode[vNode].clear();
    }

	for(ui i = 0; i < userGraphMap.size(); i++)
	{
		vector <ui> followers = userGraphMap[i];
		// cout << "At Node -- " << i << endl;
		uNode = i;

		for(int kPrimeTopic = 0; kPrimeTopic < numTopics; kPrimeTopic++)
		{
			float sumVal = 0;
			for(int kTopic = 0; kTopic < numTopics; kTopic++)
			{
				sumVal += (nodeTopicCountMap[uNode][kTopic] * topicTopicInfluence[kTopic][kPrimeTopic]);
			}
			NuNodekTopicTkTopickPrimeTopicUKPrime[uNode][kPrimeTopic] = sumVal;
		}

		for(ui j = 0; j < followers.size(); j++)
		{
			vNode = followers[j];

			double shapeParam = nodeNodeCount[uNode][vNode] + baseAlpha;

			double totalNu = 0;
			for(int kPrimeTopic = 0; kPrimeTopic < numTopics; kPrimeTopic++)
			{
				totalNu += (userTopicInfluence[vNode][kPrimeTopic] * NuNodekTopicTkTopickPrimeTopicUKPrime[uNode][kPrimeTopic]);
			}

			// double wuv = ((Nuv + baseAlpha) * 1.0)/(totalNu + baseBeta);
			double wuv = getSampleFromGamma(shapeParam, 1.0/(totalNu + baseBeta));
			
			userUserInfluence[uNode][vNode] = wuv;

			userUserInfluenceSumPerNode[uNode] += wuv;

			if(ITE >= BURN_IN && ITE%10 == 0)
			{
				userUserInfEvery10thIter[uNode][vNode] += wuv;
			}

            for(int kTopic = 0; kTopic < numTopics; kTopic++)
			{
                NuNodekTopicWuNodevNodeSumUNode[vNode][kTopic] += (nodeTopicCountMap[uNode][kTopic] * wuv);
			}

			avgVal += wuv;

			count += 1;

			// cout << uNode << " " << vNode << " " << Nuv << " " << totalNu <<  " " << wuv << endl;
			// wuvFile << uNode << " " << vNode << " " << nodeNodeCount[uNode][vNode] << " " << totalNu << " " << wuv << endl;
		}

		nodeNodeCount[uNode].clear();
	}

	nodeNodeCount.clear();

	cout << "Avg Wuv value = " << (avgVal * 1.0) / count  << " Over Edges = " << count << endl;

	if(ITE >= BURN_IN && ITE%10 == 0)
	{
		avgOverNumChains += 1;
	}

	return 0;
}

///////////////////////////////////////////////////
/////// SAMPLE TOPIC TOPIC INFLUENCE///////////////
///////////////////////////////////////////////////

int sampleTopicTopicInfluence(int ITE)
{
	unordered_map<ui, unordered_map<ui, ui> >::iterator topicTopicCountIterator;
	double avgVal = 0;
	int count = 0;
	int kTopic, kPrimeTopic;
	
	topicTopicInfluenceSumPerTopic.clear();

	for(int i = 0; i < numTopics; i++)
	{
		kTopic = i;

		// for(int uid = 0; uid < maxNumNodes; uid++)
		// {
		// 	totalTk += (topicNodeCountMap[kTopic][uid] * userUserInfluenceSumPerNode[uid]);
		// }

		for(int j = 0; j < numTopics; j++)
		{
			kPrimeTopic = j;

			// float betaVal = 1.0/topicTopicThetaParamPrior[kTopic][kPrimeTopic];
			float betaVal = 1.0/2;
			
            double totalTk = 0;
            for(int vNode = 0; vNode < maxNumNodes; vNode++)
            {
                totalTk += (userTopicInfluence[vNode][kPrimeTopic] * NuNodekTopicWuNodevNodeSumUNode[vNode][kTopic]);
            }

			double scaleParam = 1/(totalTk + betaVal);

			double shapeParam = topicTopicCount[kTopic][kPrimeTopic] + baseAlphaTopic;

			// double tkk = getSampleFromGamma(shapeParam, 1.0/(totalTk + baseBeta));
			double tkk = getSampleFromGamma(shapeParam, scaleParam);
			
			topicTopicInfluence[kTopic][kPrimeTopic] = tkk;

			topicTopicInfluenceSumPerTopic[kTopic] += tkk;

			if(ITE >= BURN_IN && ITE%10 == 0)
			{
				topicTopicInfEvery10thIter[kTopic][kPrimeTopic] += tkk;
			}

			avgVal += tkk;

			count += 1;

		}
		topicTopicCount[kTopic].clear();
	}

	topicTopicCount.clear();

	cout << "Avg Tkk value = " << (avgVal * 1.0) / count  << " Over Edges = " << count << endl;

	return 0;
}

///////////////////////////////////////////////////
/////// SAMPLE USER TOPIC INFLUENCE///////////////
///////////////////////////////////////////////////

int sampleUserTopicInfluence(int ITE)
{
	unordered_map<ui, unordered_map<ui, ui> >::iterator topicTopicCountIterator;
	double avgVal = 0;
	int count = 0;
	int vNode, kPrimeTopic;
	
	userTopicInfluenceSumPerNode.clear();

	for(int i = 0; i < numNodes; i++)
	{
		vNode = i;

		for(int j = 0; j < numTopics; j++)
		{
			kPrimeTopic = j;

			// float betaVal = 1.0/topicTopicThetaParamPrior[kTopic][kPrimeTopic];
			float betaVal = 1.0/2;
			
            double totalNuk = 0;
            for(int kTopic = 0; kTopic < numTopics; kTopic++)
            {
                totalNuk += (topicTopicInfluence[kTopic][kPrimeTopic] * NuNodekTopicWuNodevNodeSumUNode[vNode][kTopic]);
            }
			// cout << vNode << " " << kPrimeTopic << " " << totalNuk << "\n";
			double scaleParam = 1/(totalNuk + betaVal);

			// double shapeParam = nodeTopicCountMap[vNode][kPrimeTopic] + baseAlphaTopic;
			double shapeParam = nodeTopicCountToWhere[vNode][kPrimeTopic] + baseAlphaTopic;

			// double qvkPrime = getSampleFromGamma(shapeParam, 1.0/(totalNuk + baseBeta));
			double qvkPrime = getSampleFromGamma(shapeParam, scaleParam);
			
			userTopicInfluence[vNode][kPrimeTopic] = qvkPrime;

			userTopicInfluenceSumPerNode[vNode] += qvkPrime;

			if(ITE >= BURN_IN && ITE%10 == 0)
			{
				userTopicInfEvery10thIter[vNode][kPrimeTopic] += qvkPrime;
			}

			avgVal += qvkPrime;

			count += 1;

		}
		// topicTopicCount[kTopic].clear();
	}

	// topicTopicCount.clear();

	cout << "Avg QvkPrime value = " << (avgVal * 1.0) / count  << " Over Edges = " << count << endl;

	return 0;
}



///////////////////////////////
/////// SAMPLE TOPIC //////////
///////////////////////////////

int sampleTopicAssignment(int ITE)
{
	cout << "Sample Topic Assignment\n";
	int assignedTopic;

	possTopicTime = 0;

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{
		int eventIndex = newSyntheticEvents[i][0];		
		// if(emptyEvents[i] != 1)
		// {
		// cout << "Doc sizes --" << allEventsDocs[i].size() << " " << allEventsMentions[i].size() << " " << emptyEvents[i] << "\n";
		// cout << "Working on event --- " << i << endl;
		// li eventIndex = i;
		int eventNode = newSyntheticEvents[i][1];
		int eventParent = newSyntheticEvents[i][2];
		int eventTopic = newSyntheticEvents[i][3];
		int eventLevel = newSyntheticEvents[i][4];

		// if(eventLevel >= 0 && eventLevel <= maxTrainLevel)
		if(eventLevel <= maxTrainLevel)
		{
			vector<ui> currDoc = allEventsDocs[i];
			vector<ui> currMention = allEventsDocs[i];

			// cout << "Got event details  --- " << i << "---" << eventNode << " " << eventParent << " " << eventTopic <<endl;

			decreamentCountFromMatrices(eventIndex, eventNode, eventParent, eventTopic, currDoc, currMention, 1);
			// cout << "Done decreamentiing..." << endl;
			assignedTopic =  getSampledTopicAssignment(eventIndex, eventNode, eventParent, currDoc, currMention, ITE);
		
			increamentCountToMatrices(eventIndex, eventNode, eventParent, assignedTopic, currDoc, currMention, 1);

			newSyntheticEvents[eventIndex][3] = assignedTopic;
			// }
			// if the event content is not present, topic assignment should happen depending on event generation time... 
			// else
			// {
			// 	assignedTopic = numTopics + 1;
			// 	newSyntheticEvents[eventIndex][3] = assignedTopic;
			// }

			// if(i % 1000 == 0)
			// {
			// 	cout << "Topic Assignment -- " << i << endl;
			// 	cout << "possible topic time -- " << possTopicTime << endl;
			// }
		}
		else
		{
			assignedTopic = invalidTopicId;
			newSyntheticEvents[eventIndex][3] = assignedTopic;
		}

		if(ITE >= BURN_IN && ITE%10 == 0)
		{
			topicDistEvery10thIter[eventIndex].push_back(assignedTopic);
		}
	}

	// printTopicTopicCount();

	return 0;
}

int getSampledTopicAssignment(int eventIndex, int eventNode, int eventParent, vector<ui> currDoc, vector<ui> currMention, int ITE)
{
	int assignedTopicInd, assignedTopic;

	int eventParentTopic, eventParentIndex, eventParentNode;

	double eventGenerationBRorTrigger;
	vector<double> topicAssignmentScores;

	// for(ui i = 0; i < possibleSetOfTopics.size(); i++)
	for(int topicId = 0; topicId < numTopics; topicId++)
	{
		// int topicId = possibleSetOfTopics[i];
		eventParentIndex = eventParent;

		high_resolution_clock::time_point t1, t2;
		auto duration = 0;

		t1 = high_resolution_clock::now();
		if(eventParentIndex > -1)
		{
			eventParentTopic = newSyntheticEvents[eventParentIndex][3];
			eventParentNode = newSyntheticEvents[eventParentIndex][1];

			double eventTriggerTerm = getEventGenerationTerm(eventParentIndex, eventParentNode, eventParentTopic, eventIndex, eventNode, topicId);

			eventGenerationBRorTrigger = eventTriggerTerm;
		}
		else
		{
			double baseRateTerm = getBaseRateTerm(topicId, eventNode);
			eventGenerationBRorTrigger = baseRateTerm;
		}
		t2 = high_resolution_clock::now();
		duration = duration_cast<microseconds>( t2 - t1 ).count();

		eGenTermTime += duration;
		// 4 terms...
		// cout << "Getting topic mention term --- " << endl;
		// double topicMentionTerm = getTopicMentionTerm(topicId, eventIndex, currMention);
		// cout << "Got topic mention term --- " << endl;
		duration = 0;
		t1 = high_resolution_clock::now();
		double topicWordTerm = getTopicWordTerm(topicId, eventIndex, currDoc);
		t2 = high_resolution_clock::now();
		duration = duration_cast<microseconds>( t2 - t1 ).count();

		twTermTime += duration;

		// cout << "Got topic word term --- " << endl;
		// double childTopicTopicTerm = getChildTopicTopicTerm(eventIndex, topicId);
		// cout << "Got child topic term --- " << endl;
		duration = 0;
		t1 = high_resolution_clock::now();
		double childEventsTerm = getChildEventsTerm(eventIndex, topicId, eventNode);
		t2 = high_resolution_clock::now();
		duration = duration_cast<microseconds>( t2 - t1 ).count();

		chGenTermTime += duration;
		// double score = userTopicOrTopicTopicTerm + topicMentionTerm + topicWordTerm + childTopicTopicTerm;
		double score = eventGenerationBRorTrigger + topicWordTerm + childEventsTerm;
		// cout << score << " ";

		// if(eventIndex == 5 || eventIndex == 7)
		// {
		// 	cout << topicId << " " << eventGenerationBRorTrigger << " " << topicWordTerm << " " << childEventsTerm << "\n";
		// }

		topicAssignmentScores.push_back(score);
	}

	// if(eventIndex == 5 || eventIndex == 7)
	// {
	// 	cout << "eventId --" << eventIndex << " ";
	// 	for(int topicId = 0; topicId < numTopics; topicId++)
	// 	{
	// 		cout << topicAssignmentScores[topicId] << ", ";
	// 	}
	// 	cout << "\n";
	// }

	// cout << "\n";
	// score for additional topic... this is just based on the priors...
	// double additionalTopicScore = getScoreForAdditionalTopic();

	assignedTopicInd =  getSampleFromMultinomial(topicAssignmentScores);
	// cout << "Assigned Topic Index = " << assignedTopicInd << endl;

	// assignedTopic = possibleSetOfTopics[assignedTopicInd];
	assignedTopic = assignedTopicInd;

	// cout << "Assigned Topic = " << assignedTopic << endl;

	// if(ITE > BURN_IN && ITE%10 == 0)
	// {
	// 	for(ui i = 0; i < avgTopicProbVector[eventIndex].size(); i++)
	// 	{
	// 		avgTopicProbVector[eventIndex][i] += calculatedProbVec[i];
	// 	}
	// }

	return assignedTopic;
}

double getEventGenerationTerm(int eventParentIndex, int eventParentNode, int eventParentTopic, int eventIndex, int eventNode, int topicId)
{
	double logFinalFirstTerm = 0.0;
	double logSurvivalTerm, logHazardTermTD;

	double timeDiff = eventIndexTimestamps[eventParentIndex] - eventIndexTimestamps[eventIndex]; // this is negative -- parentTime is less...
	double hazardTerm = userUserInfluence[eventParentNode][eventNode] * topicTopicInfluence[eventParentTopic][topicId] * userTopicInfluence[eventNode][topicId];

	if(hazardTerm > 0)
	{
		logSurvivalTerm = -(hazardTerm);
		logHazardTermTD = log(hazardTerm) + timeDiff;
		logFinalFirstTerm = logSurvivalTerm + logHazardTermTD;
	}
	
	return logFinalFirstTerm;
}

double getBaseRateTerm(int topicId, int eventNode)
{
	double finalBRTerm = 0.0;
	double hazardTerm = 0.0, logHazardTerm = 0.0, logSurvivalTerm = 0.0;

	hazardTerm = userBaseRateMap[eventNode] * topicBaseRateMap[topicId];
	logHazardTerm = log(hazardTerm);
	logSurvivalTerm = -(hazardTerm * totalTime);
	
	finalBRTerm = logHazardTerm + logSurvivalTerm;

	return finalBRTerm;
}

double getChildEventsTerm(int eventIndex, int topicId, int eventNode)
{
	// unordered_map <int, ui> topicCounts;

	double finalChildEventTerm = 0.0;

	vector <ui> childEventsList;
	try
	{
		childEventsList =  childEventsMap.at(eventIndex);
	}
	catch(exception &e)
	{
		// return topicCounts;								// returning empty dict/map
		return finalChildEventTerm;							// returning zero contribution from child events term..
	}

	if(childEventsList.size() > 0)
	{
		int childEventIndex, childEventNode, childEventTopic;
		double eachChildEventTerm;

		for(unsigned int i = 0; i < childEventsList.size(); i++)
		{
			childEventIndex = childEventsList[i];
			childEventNode = newSyntheticEvents[childEventIndex][1];
			childEventTopic = newSyntheticEvents[childEventIndex][3];

			eachChildEventTerm = getEventGenerationTerm(eventIndex, eventNode, topicId, childEventIndex, childEventNode, childEventTopic);

			finalChildEventTerm += eachChildEventTerm;
		}
	}

	return finalChildEventTerm;
}


double getTopicWordTerm(int topicId, int eventIndex, vector<ui> currDoc)
{
	vector<ui> wordHistVec = wordHistAllDocsVector[eventIndex];

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
		// double baseTerm = alphaWord + NTWCountVec[topicId][word];
		double baseTerm = alphaTopicWordPrior + NTWCountVec[topicId][word];

		double valueForWord = 1.0;

		for(ui j = 0; j < wordCount; j++)
		{
			valueForWord *= (baseTerm + j);
		}

		finalThirdValueNume *= valueForWord;
		i++;
	}

	double baseTermDenom = sumAlphaTopicWordPrior + NTWSumWordsVec[topicId];

	for(ui i = 0; i < currDoc.size(); i++)
	{
		finalThirdValueDenom *= (baseTermDenom + i);
	}

	finalThirdValue = finalThirdValueNume / finalThirdValueDenom;

	finalThirdValue = log(finalThirdValue);

	return finalThirdValue;
}


////////////////////////////////
/////// SAMPLE PARENT //////////
////////////////////////////////

int sampleParentAssignment(int ITE)
{
	cout << "Sampling Parent Assignment ... \n";

	// int assignedParentNode;

	for(unsigned int i = 0; i < newSyntheticEvents.size(); i++)
	{
		int eventIndex = newSyntheticEvents[i][0];

		int eventNode = newSyntheticEvents[i][1];
		// int eventParent = newSyntheticEvents[i][2];
		int eventTopic = newSyntheticEvents[i][3];

		int eventLevel = newSyntheticEvents[i][4];

		int assignedParent;

		// if(eventLevel >= 0 && eventLevel <= maxTrainLevel)
		if(eventLevel <= maxTrainLevel)
		{
			
			double eventTime = eventIndexTimestamps[i];

			vector<ui> currDoc = allEventsDocs[i];
			// vector<ui> currMention = allEventsMentions[i];

			// decreamentCountFromMatrices(eventIndex, eventNode, eventParent, eventTopic, currDoc, currMention, 0);
			assignedParent = getSampledParentAssignment(eventTime, eventNode, eventIndex, eventTopic, ITE);
			// increamentCountToMatrices(eventIndex, eventNode, assignedParent, eventTopic, currDoc, currMention, 0);

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
			parentDistEvery10thIter[eventIndex].push_back(assignedParent);
		}

	}

	cout << "\nLog Likelihood After Parent Assignment = " << logLikelihood << "\n";

	return 0;
}


int getSampledParentAssignment(double eventTime, li eventNode, li eventIndex, li eventTopic, int ITE)
{
	int assignedParent;

	vector <ui> possibleParentEvents = allPossibleParentEvents[eventIndex];
	vector<double> possibleParentExp = allPossibleParentEventsExponentials[eventIndex];
	
	vector <double> calculatedProbVec;
	vector <double> newCalculatedProbVec;

	if(possibleParentEvents.size() > 0)
	{
		calculatedProbVec = populateCalculatedProbVec(possibleParentEvents, possibleParentExp, eventNode, eventTopic, eventTime, ITE);

		ui sampledIndex = getSampleFromDiscreteDist(calculatedProbVec);

		if(sampledIndex == calculatedProbVec.size()-1)
		{
			assignedParent = -1;
		}
		else
		{
			assignedParent = possibleParentEvents[sampledIndex];
		}

		// cout << "sampled parent evaluated --- " << assignedParent << endl; 
		logLikelihood += log(calculatedProbVec[sampledIndex]);
	}
	else
	{
		assignedParent = -1;
		calculatedProbVec.push_back(1);
	}

	// updating for avgProbability of the parent events... at the end lets write it to a file...  
	if(ITE > BURN_IN && ITE%10 == 0)
	{
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
	vector <double> calculatedProbVec(possibleParentEvents.size() + 1, 0.0);
	
	// double firstTermNume, firstTermDenom, firstTerm, secondTerm;
	double hazFinalTerm = 0;

	double normalizationFactor = 0;

	unsigned int k;

	for(k = 0; k < possibleParentEvents.size(); k++)
	{
		vector <int> possParentEvent = newSyntheticEvents[possibleParentEvents[k]];

		int possParentNode = possParentEvent[1];
		int possParentEventTopic = possParentEvent[3];

		if(possParentEventTopic != invalidTopicId)
		{
			double Wuv, Taukk, Qouk;
			Wuv = userUserInfluence[possParentNode][eventNode];
			Taukk = topicTopicInfluence[possParentEventTopic][eventTopic];
			Qouk = userTopicInfluence[eventNode][eventTopic];

			if(Wuv > 0 && Taukk > 0)
			{
				double temporalDecay = possibleParentExp[k];
				double uuTTProdTerm = Wuv * Taukk * Qouk;
				double infHazardTerm = uuTTProdTerm * temporalDecay;
				double infSurvivalTerm = exp(-(uuTTProdTerm));
				hazFinalTerm = infSurvivalTerm * infHazardTerm;
			}

			calculatedProbVec[k] = hazFinalTerm;
		}
		else
		{
			calculatedProbVec[k] = 0;	
		}
		normalizationFactor += calculatedProbVec[k];
	}

	// get prob of having no parent

	double brFinalTerm = 0.0;
	double brHazardTerm = userBaseRateMap[eventNode] * topicBaseRateMap[eventTopic];
	double brSurvivalTerm = exp(-(brHazardTerm * totalTime));

	brFinalTerm = brSurvivalTerm * brHazardTerm;

	calculatedProbVec[k] = brFinalTerm;
	normalizationFactor += calculatedProbVec[k];

	for(ui i = 0; i < calculatedProbVec.size(); i++)
	{
		calculatedProbVec[i] = calculatedProbVec[i] / normalizationFactor;
		// cout << calculatedProbVec[i] << " ";
	}
	// cout << "\n";
	return calculatedProbVec;
}




///////////////////////////////
/////// User Base Rates  //////
///////////////////////////////

int updateUserBaseRates()
{
	int sponCount = 0;
	int minTime = eventIndexTimestamps[0];
	int maxTime = eventIndexTimestamps[eventIndexTimestamps.size() - 1];

	int totalDataObservedTime = (maxTime - minTime) / timeScalingFactor;
	// int totalDataObservedTime = (maxTime - minTime);

	map<int, int> eachNodeSponCount;
	// map<int, bool> distinctNodes;

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{
		// if(newSyntheticEvents[i][4] >= 0 && newSyntheticEvents[i][4] <= maxTrainLevel)
		if(newSyntheticEvents[i][4] <= maxTrainLevel)
		{
			if(newSyntheticEvents[i][2] == -1)
			{
				sponCount++;
				// distinctNodes[newSyntheticEvents[i][1]] = 1;
				eachNodeSponCount[newSyntheticEvents[i][1]]++;
			}
		}
	}

	double totalTopicBaseRate = 0.0;

	map<ui, double>::iterator topicBaseRateMapIt;

	for(topicBaseRateMapIt = topicBaseRateMap.begin(); topicBaseRateMapIt != topicBaseRateMap.end(); topicBaseRateMapIt++)
	{
		double tbr = topicBaseRateMapIt->second;
		totalTopicBaseRate += tbr;
	}

	double averageMuVal = 0;

	double minMuVal = 1000000;
	double maxMuVal = 0;
	map<int, int>::iterator eachNodeSponCountIt;

	for(eachNodeSponCountIt = eachNodeSponCount.begin(); eachNodeSponCountIt != eachNodeSponCount.end(); eachNodeSponCountIt++)
	{
		int nodeId = eachNodeSponCountIt->first;
		int nodeSponCount = eachNodeSponCountIt->second;

		userBaseRateMap[nodeId] = (nodeSponCount * 1.0) / (totalDataObservedTime * totalTopicBaseRate);

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
/////// Topic Base Rates  /////
///////////////////////////////

int updateTopicBaseRates()
{
	int sponCount = 0;
	int minTime = eventIndexTimestamps[0];
	int maxTime = eventIndexTimestamps[eventIndexTimestamps.size() - 1];

	int totalDataObservedTime = (maxTime - minTime) / timeScalingFactor;
	// int totalDataObservedTime = (maxTime - minTime);

	map<int, int> eachTopicSponCount;
	// map<int, bool> distinctTopics;

	for(ui i = 0; i < newSyntheticEvents.size(); i++)
	{
		// if(newSyntheticEvents[i][4] >= 0 &&  newSyntheticEvents[i][4] <= maxTrainLevel)
		if(newSyntheticEvents[i][4] <= maxTrainLevel)
		{
			if(newSyntheticEvents[i][2] == -1)
			{
				sponCount++;
				eachTopicSponCount[newSyntheticEvents[i][3]]++;
			}
		}
	}

	double totalUserBaseRate = 0.0;

	map<ui, double>::iterator userBaseRateMapIt;

	for(userBaseRateMapIt = userBaseRateMap.begin(); userBaseRateMapIt != userBaseRateMap.end(); userBaseRateMapIt++)
	{
		double ubr = userBaseRateMapIt->second;
		totalUserBaseRate += ubr;
	}

	double averageMuVal = 0;

	double minMuVal = 1000000;
	double maxMuVal = 0;
	map<int, int>::iterator eachTopicSponCountIt;

	for(eachTopicSponCountIt = eachTopicSponCount.begin(); eachTopicSponCountIt != eachTopicSponCount.end(); eachTopicSponCountIt++)
	{
		int topicId = eachTopicSponCountIt->first;
		int topicSponCount = eachTopicSponCountIt->second;

		topicBaseRateMap[topicId] = (topicSponCount * 1.0) / (totalDataObservedTime * totalUserBaseRate);

		averageMuVal +=  topicBaseRateMap[topicId];

		if(topicBaseRateMap[topicId] > maxMuVal)
		{
			maxMuVal = topicBaseRateMap[topicId];
		}

		if(topicBaseRateMap[topicId] < minMuVal)
		{
			minMuVal = topicBaseRateMap[topicId];
		}
	}

	defaultMuVal = averageMuVal / eachTopicSponCount.size();
	defaultTopicBaseRate = averageMuVal / eachTopicSponCount.size();
	
	cout << "sponCount = " << sponCount << " minMuVal = " << minMuVal << " maxMuVal = " << maxMuVal << " MLE MuVal = " << defaultMuVal << " " << defaultTopicBaseRate << endl;

	return 0;
}


///////////////////////////////
////////   VALIDATIONS   ///////
///////////////////////////////


int writeEvery10ItersTopicAssignmentToFile()
{
	ofstream topicAssignmentsFile;
	// topicAssignmentsFile.open("B5_topicAssignments_OurModel_estAll.txt");
	topicAssignmentsFile.open("topicAssignments.txt");

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
	// parentAssignmentsFile.open("B5_parentAssignments_OurModel_estAll.txt");
	parentAssignmentsFile.open("parentAssignments.txt");

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
	// avgProbVecFile.open("B5_avgParProb_OurModel_estAll.txt");
	avgProbVecFile.open("parentAssignmentAvgProbFile.txt");

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

int writeTopicTopicInfluenceToFile()
{
	ofstream tkkinfFile, tkkinfFileAvg;
	tkkinfFile.open("topicTopicInf.txt");
	tkkinfFileAvg.open("topicTopicInfAvg.txt");

	cout << "Writing topic-topic influences...\n";

	int kTopic, kPrimeTopic;

	for(int i = 0; i < numTopics; i++)
	{
		kTopic = i;

		tkkinfFile << numTopics << " " << kTopic << " ";
		tkkinfFileAvg << numTopics << " " << kTopic << " ";

		for(int j = 0; j < numTopics; j++)
		{
			kPrimeTopic = j;
			tkkinfFile << kPrimeTopic << " " << topicTopicInfluence[kTopic][kPrimeTopic] << " ";
			tkkinfFileAvg << kPrimeTopic << " " << topicTopicInfEvery10thIter[kTopic][kPrimeTopic] / 10 << " ";
		}

		tkkinfFile << "\n";
		tkkinfFileAvg << "\n";
	}

	tkkinfFile.close();
	tkkinfFileAvg.close();

	return 0;
}

int writeUserTopicInfluenceToFile()
{
	ofstream uukinfFile, uukinfFileAvg;
	uukinfFile.open("userTopicInf.txt");
	uukinfFileAvg.open("userTopicInfAvg.txt");

	cout << "Writing topic-topic influences...\n";

	int uNode, kTopic;

	for(int i = 0; i < numNodes; i++)
	{
		uNode = i;

		uukinfFile << numTopics << " " << uNode << " ";
		uukinfFileAvg << numTopics << " " << uNode << " ";

		for(int j = 0; j < numTopics; j++)
		{
			kTopic = j;
			uukinfFile << kTopic << " " << userTopicInfluence[uNode][kTopic] << " ";
			uukinfFileAvg << kTopic << " " << userTopicInfEvery10thIter[uNode][kTopic] / 10 << " ";
		}

		uukinfFile << "\n";
		uukinfFileAvg << "\n";
	}

	uukinfFile.close();
	uukinfFileAvg.close();

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



int writeTopicBaseRatesToFile()
{

	ofstream ubrFile;
	ubrFile.open("topicBaseRate.txt");

	for(int i = 0; i < maxNumTopics; i++)
	{
		int uid = i;
		double ubrUid = topicBaseRateMap[i];

		if(ubrUid == 0)
		{
			ubrUid = defaultTopicBaseRate;	
		}

		// ubrFile << uid << " " << ubrUid << "\n";
		ubrFile << setprecision(10) << uid << " " << ubrUid << "\n";
	}

	ubrFile.close();

	return 0;
}

///////////////////////////////
/////// UTILITY FUNCTIONS /////
///////////////////////////////

int getSampleFromMultinomial(vector<double> calculatedProbVec)
{
	int assignedInd;

	// get normalized prob vector...
	vector <double> normalizedProbVector = getNormalizedLogProb(calculatedProbVec);
	
	// for(int i = 0; i < normalizedProbVector.size(); i++)
	// {
	// 	cout << normalizedProbVector[i] << " ";
	// }
	// cout << "\n";
	assignedInd = getSampleFromDiscreteDist(normalizedProbVector);

	// cout << "Assigned Index -- " << assignedInd << endl;

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


// Decreament and Increament Counters
int decreamentCountFromMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> currDoc, vector<ui> currMention, bool topicSampling)
{
	// cout << "decreament count from topic-topic matrix.. from the cell parentTopic -> eventTopic" << endl;
	// decreament only if the eventTopic is valid...
	if(eventTopic >= 0)
	{

		if(topicSampling == 1)
		{
			// decreament the count of mentions corresponding to this event...
			// Dirichlet Tree Prior
			// decreamentCountFromTopicMentions(eventTopic, currMention);

			// decreament the count of words corresponding to this event...
			// Dirichlet Distribution Decreament
			decreamentCountFromTopicWord(eventTopic, currDoc);
		}
	}

	return 0;
}


int increamentCountToMatrices(int eventIndex, int eventNode, int eventParent, int eventTopic, vector <ui> currDoc, vector<ui> currMention, bool topicSampling)
{
	// cout << "increament count to topic-topic matrix... in the cell parentTopic -> assignedTopic" << endl;

	if(topicSampling == 1)
	{
		// decreament the count of mentions corresponding to this event...
		// Dirichlet Tree Prior
		// increamentCountInTopicMentions(eventTopic, currMention);

		// increament counts in topic-word...
		// cout << eventIndex << endl;
		increamentCountInTopicWord(eventTopic, currDoc);
		// printUnorderedMap("psitopic");
	}

	return 0;
}


int decreamentCountFromTopicWord(int eventTopic, vector <ui> currDoc)
{
	// sum of of the freq of all words for each topic... 
	if(NTWSumWordsVec[eventTopic] >= (int)currDoc.size())
	{
		NTWSumWordsVec[eventTopic] -= (int)currDoc.size();
	}
	else
	{
		// cout << "Somethings wrong with the Topic Word Counts... \n";
		NTWSumWordsVec[eventTopic] = 0;
	}

	// decreament freq of each word corresponding to the topic...
	for(ui i = 0; i < currDoc.size(); i++)	
	{
		if(NTWCountVec[eventTopic][currDoc[i]] > 0)
		{
			NTWCountVec[eventTopic][currDoc[i]] -= 1;
		}
	}

	return 0;
}


int increamentCountInTopicWord(int eventTopic, vector<ui> currDoc)
{
	NTWSumWordsVec[eventTopic] += currDoc.size();
	// NTSumWords[eventTopic] += doc.size();

	if(NTWSumWordsVec[eventTopic] > totalWords)
	{
		cout << "Total words = " << totalWords << " NTW sum = " << NTWSumWordsVec[eventTopic] << "\n";
		cout << "Sum issue with the total number of words...\n";
		exit(0);
	}

	for(unsigned int i = 0; i < currDoc.size(); i++)
	{
		NTWCountVec[eventTopic][currDoc[i]] += 1;

		if(NTWCountVec[eventTopic][currDoc[i]] > totalWords)
		{
			cout << "Total words = " << totalWords << " NTW = " << NTWCountVec[eventTopic][currDoc[i]] << " " << currDoc[i] << "\n";
			exit(0);
		}
	}
	
	return 0;
}


// Histogram of words for all Docs...
int createWordHistForAllDocs()
{
	for(unsigned int i = 0; i < allEventsDocs.size(); i++)
	{
		vector<ui> currDoc = allEventsDocs[i];

		unordered_map<ui, ui> wordHist;
		unordered_map<ui, ui>::iterator wordHistIt;
		wordHist = getHistOfWordsOverWordsFromDoc(currDoc);

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
		li word = doc[i];

		wordCounts[word] += 1;
	}

	return wordCounts;
}


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

			/*
			if(eventParent == -1)
			{
				cout << line << endl;
				cout << "eventIndexTimestamps[i] = " << eventIndexTimestamps[i] << endl; 
				cout << " Spon event -- " << eventTime << " " << eventNode << " " << eventParent << " " << eventTopic << endl;
			}
			*/
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
	vector < vector <ui> > allDocsLocal;

	double avgDocLength = 0;

	ifstream allDocsFile;
    allDocsFile.open(fileName);

    string line;
    stringstream ss;
    li i = 0;

    if(allDocsFile.is_open())
    {
	    while(getline(allDocsFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;

			ui word;
			vector <ui> currDoc;
			while(ss >> word)
			{
				// vector<double> tempEvent;
				currDoc.push_back(word);
			}
			
			// eventIndexDocSizeMap[i] = currDoc.size();
			i++;

			allDocsLocal.push_back(currDoc);
			line.clear();

			avgDocLength += currDoc.size();

			// if(i > 100000)
				// break;
		}
		allDocsFile.close();

		totalWords = avgDocLength;
		cout << "Total Number of words = " << totalWords << "\n";

		avgDocLength = avgDocLength/allDocsLocal.size();
		
		cout << "Read " << allDocsLocal.size() << " documents\n";
		cout << "Avg Doc Length = " << avgDocLength << endl;
    }
    else
    {
    	cout << "Error opening file -- " << fileName << "\n";
    	exit(0);
    }
    cout << "read " << allDocsLocal.size() << " docs\n";
	return allDocsLocal;
}

vector < vector <ui> > readIntVectorMapFromFile(string fileName)
// unordered_map <ui, unordered_map<ui, int> > readIntVectorMapFromFile(string fileName)
{
	ifstream intVectorFile;
	intVectorFile.open(fileName);
	// intVectorFile.open("mapped_users_followers_graph_sorted_1000000_users.txt");

	// vector < vector <ui> > intVectorMap;
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


// vector < vector <ui> > populateThetaPriorForTopicPairs(string fileName)
unordered_map <int, unordered_map<int, float> > populateThetaPriorForTopicPairs(string fileName)
{
	ifstream intVectorFile;
	intVectorFile.open(fileName);
	// intVectorFile.open("mapped_users_followers_graph_sorted_1000000_users.txt");

	// vector < vector <ui> > intVectorMap;
	unordered_map <int, unordered_map<int, float> > localTTParam(numTopics);

	string line;
	stringstream ss;

	ui count; 
	ui mapsize = 0;
	int kTopic, kPrimeTopic;
	float ttval;

	if (intVectorFile.is_open())
	{
		while(getline(intVectorFile, line))
		{
			ss.clear();
			ss.str("");

			ss << line;
			ss >> count >> kTopic;

			// vector <ui> tempVec;
			unordered_map<int, float> tempMap;

			for(unsigned i = 0; i < count; i++)
			{
				ss >> kPrimeTopic >> ttval;
				tempMap[kPrimeTopic] = ttval;
			}

			localTTParam[kTopic] = tempMap;

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
	return localTTParam;
}



int populateListOfEmptyEvents()
{
	// cout << "List of empty events --- ";
	for(ui i = 0; i < allEventsDocs.size(); i++)
	{
		// if(allEventsDocs[i].size() == 0 && allEventsMentions[i].size() == 0)
		if(allEventsDocs[i].size() == 0)
		{
			emptyEvents[i] = 1;
			cout << i << " ";
		}
	}
	// cout << "\n";
	cout << "Empty Events -- " << emptyEvents.size() << endl;

	return  0;
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
