# ReadMe

To generate data, run "createData.sh" situated in centralFiles.
"createData.sh" runs two code files.
(1) "generateData_dnhp.py" - To generate synthetic data as per the Dual Network Hawkes Process (DNHP)
	The input files required are:
		(a) User-User graph: each line in the file is described as follows:
			NumOfNeighbors User-Id Neighbor-1 <space> Neighbor-2 <space> Neighbor-3 ... <space> Neighbor-NumOfNeighbors
		(b) Topic-Topic graph (has similar format as user-user graph)
		(c) User-User influence 
			NumOfNeighbors User-Id Neighbor-1 <space> InfluenceOnNeighbor-1 <space> Neighbor-2 <space>  InfluenceOnNeighbor-2 <space>  ... <space> Neighbor-NumOfNeighbors <space>InfluenceOnNeighbor-NumOfNeighbors
		(d) Topic-Topic influence (has similar format as user-user influence file)
		(e) User base rates: Each line is of the form:
			UserId BaseRate
		(f) Topic base rates: (has similar format as user base rate file)
		
	This code also generates the topic-word distribution file.

	The final output of this is two files (a) Events File - "events_1L.txt" and (b) Documents File - "docs_1L.txt"

(2) "getStoreTopKCandParents.cpp" - To generate the list of top-100 candidate parents for each event.
	This takes in the events file as the parameter and reverse user-user graph as input.
	This code outputs two files (a) "top100Parents.txt"  and (b) "top100ParentExps.txt"
	(a) "top100Parents.txt" for each event consists of maximum 100 candidate parents ordered as nearest in time first order.
		The format for the same is:
		NumCandidateParents <space> EventId <space> CandidateParentEventId-1 <space> CandidateParentEventId-2 <space> ... <space> CandidateParentEventId-100

	(b) "top100ParentExps.txt" for each event consists of maximum 100 candidate parents ordered as nearest in time first order.
		The format for the same is:
		NumCandidateParents <space> EventId <space> exp(Time Diff with CandidateParentEventId-1) <space> (Time Diff with CandidateParentEventId-2) <space> ... <space> (Time Diff with CandidateParentEventId-100)



Once the data is generated, you are good to run the inference models now.
The code for each model is in the respective folder.
Each of the models folder have a "evaluateAll.sh" file that runs the inference model and evaluations after that.
The inference code runs total 300 iterations where BURN-IN period is 200 iterations and samples are taken at every 10 iteration after 200 iteration.
This inference code generates a number of files -- 
(a) estimated user-user influence (userUserInf.txt)
(b) estimated topic-topic influence (topicTopicInf.txt)
(c) parent assignment samples (parentAssignments.txt)
(d) topic assignment samples (topicAssignments.txt)
(e) average distribution over candidate parent assignments (parentAssignmentAvgProbFile.txt)


With the above generated files the evaluation is carried out.
As the topicAssignments.txt and parentAssignments.txt consists of 10 samples taken at every 10th iteration, we calculate the mode of 10 samples and assign the mode topic or mode parent for each event (using "getModeValueFromAssignments.cpp")

The code files "accPredParents.cpp" and "precAtKParentAssignment.cpp" are specifically to evaluate the parent assignment done by the model.
"accPredParents.cpp" calculates the accuracy of parent assignment using the "modeParentAssignment.txt" file, and the "precAtKParentAssignment.cpp" uses the "parentAssignmentAvgProbFile.txt" in which the candidate parents are ordered in decreasing order of probabilities, giving ranking of the candidate parents.
"precAtKParentAssignment.cpp" uses this ranking and calculates the recall at 1,3,5,7,10 and so on (results are stored in "recallAtKParent.txt").

Using the "modeTopicAssignment.txt" where we have the estimated topic assignment for each event the word distribution is estimated using this mode topic assignment (getPredTopicWordDists.py).
Once the topic distribution is in place, the topic evaluation is carried out using "getTAETopicAssignments.py" as described in the paper in the Evaluation Tasks section.

Finally the user-user influence evaluated as described in the paper using "getEffectiveUVInfUnified.py" and the results are stored in "wuvEvaluation.txt".
