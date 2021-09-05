#!/bin/bash

shopt -s expand_aliases
source ~/.bash_aliases

echo 'compiling model...\n'
g+++ dnhp dnhp_Real_Train_newSplitTrainTest
echo 'Running for 300 iterations...\n'

touch programSummary.out
time ./dnhp | tee programSummary.out

echo "Done running the inference model..."

g+++ modeAssign getModeValueFromAssignments.cpp
./modeAssign topicAssignments.txt modeTopicAssignment.txt 
./modeAssign parentAssignments.txt modeParentAssignment.txt

echo "Getting topic word distributions"
python getPredTopicWordDists.py


echo "Estimating the test LL with content and without content..."
python -u calculateTestLLUnified_RealData_newSplitTrainTest.py | tee testLL.out
python -u calculateTestLLUnified_RealData_newSplitTrainTest_WO_Content.py | tee testLLWO.out
echo "Got LL for the Test set..."