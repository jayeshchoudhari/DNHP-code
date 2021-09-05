#!/bin/bash

shopt -s expand_aliases
source ~/.bash_aliases

echo "Compile and Run the model on the train set"
g+++ h100-1 hmhpModel_Real_Train_newSplitTrainTest.cpp
time ./h100-1 | tee programSummary.out
echo "Learned the required parameters"

echo "Would use the parameters now to estimate the LL"
g+++ modeAssign getModeValueFromAssignments.cpp
./modeAssign topicAssignments.txt modeTopicAssignment.txt 
./modeAssign parentAssignments.txt modeParentAssignment.txt

echo "Getting topic-word and user-topic distributions"
python getPredTopicWordDists.py
python getPredUserTopicDist.py

echo "Estimating LL"
python -u calculateTestLLHMHP_RealData_newSplitTrainTest.py | tee testLL.out
echo "Done"