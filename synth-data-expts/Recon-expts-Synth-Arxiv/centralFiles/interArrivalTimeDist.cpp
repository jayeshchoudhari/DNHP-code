/*
	Check if top 100 parents work...
*/

#include <iostream>
#include <cstdio>
// #include <boost/algorithm/string.hpp> 
// #include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

using ui = unsigned int;


int main(int argc, char *argv[])
{
	vector<int> eidParents;

	ifstream origEventsFile;
	origEventsFile.open("events_1L.txt");

	string line;
	stringstream ss;

	double etime;

	int uid, pid, topid, eid;

	ofstream timediffFile;
	timediffFile.open("timeDiffs.txt");

	vector<double> timeVals;

	int linenum = 0;

	while(getline(origEventsFile, line))
	{
		ss.clear();
		ss.str("");

		ss << line;

		ss >> etime >> uid >> pid >> topid;

		timeVals.push_back(etime);
		// eidParents.push_back(pid);

		if(pid > -1)
		{
			timediffFile << etime - timeVals[pid] << " " << linenum - pid  << "\n";
		}

		linenum++;
	}


	timediffFile.close();
	origEventsFile.close();

	
	return 0;
}
