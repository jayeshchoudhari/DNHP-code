#!/bin/bash

shopt -s expand_aliases
source ~/.bash_aliases

echo "Compile and Run the model on the train set"
g+++ nh100-1 netHwks_Real_Train_newSplitTrainTest.cp
time ./nh100-1 | tee programSummary.out
echo "Learned the required parameters"

echo "Would use the parameters now to estimate the LL"
g+++ modeAssign getModeValueFromAssignments.cpp
# ./modeAssign topicAssignments.txt modeTopicAssignment.txt 
./modeAssign parentAssignments.txt modeParentAssignment.txt

# python getPredTopicWordDists.py
# python getPredUserTopicDist.py

python -u calculateTestLLNetHwks_RealData_newSplitTrainTest_WO_Content.py | tee testLLWO-NH.out
