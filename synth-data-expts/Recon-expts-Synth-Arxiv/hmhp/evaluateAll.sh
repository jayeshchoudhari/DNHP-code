#!/bin/bash

shopt -s expand_aliases
source ~/.bash_aliases

echo 'compiling model...\n'
g+++ uniAll unifiedModelInference_synthData_estAll_LessCountNu.cpp
echo 'Running mode for 300 iterations...\n'

touch programSummary.out
time ./uniAll | tee programSummary.out

echo 'compiling mode prec and acc files...\n'
g+++ modeAssign getModeValueFromAssignments.cpp
g+++ precAtK precAtKParentAssignment.cpp
g+++ accpred accPredParents.cpp
echo 'Done compilaing...\n'

echo 'Getting mode files...\n'
./modeAssign parentAssignments.txt modeParentAssignment.txt
./modeAssign topicAssignments.txt modeTopicAssignment.txt
./precAtK parentAssignmentAvgProbFile.txt recallAtKParentAssignment.txt

echo 'getting pred topic distributions...'
python getPredTopicWordDists.py
echo 'Topic Assignment Evaluation..\n'
python getTAETopicAssignments.py > topicEvaluation.txt

echo 'Accuracy\n'
touch recallAtKParent.txt

./accpred modeParentAssignment.txt > recallAtKParent.txt
# echo 'Recall @1\n' >> recallAtKParent.txt
cut -d' ' -f2 recallAtKParentAssignment.txt | sort -g | uniq -c | awk '{print $2, $1/100000}' | tail -n 1 >> recallAtKParent.txt
# echo 'Recall @3\n' >> recallAtKParent.txt
cut -d' ' -f3 recallAtKParentAssignment.txt | sort -g | uniq -c | awk '{print $2, $1/100000}' | tail -n 1 >> recallAtKParent.txt
# echo 'Recall @5\n' >> recallAtKParent.txt
cut -d' ' -f4 recallAtKParentAssignment.txt | sort -g | uniq -c | awk '{print $2, $1/100000}' | tail -n 1 >> recallAtKParent.txt
# echo 'Recall @7\n' >> recallAtKParent.txt
cut -d' ' -f5 recallAtKParentAssignment.txt | sort -g | uniq -c | awk '{print $2, $1/100000}' | tail -n 1 >> recallAtKParent.txt


echo 'Wuv Error...'
python getEffectiveUVInfHMHP.py > wuvEvaluation.txt
