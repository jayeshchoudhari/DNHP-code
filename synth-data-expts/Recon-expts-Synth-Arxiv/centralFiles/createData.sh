#!/bin/bash

shopt -s expand_aliases
source ~/.bash_aliases

echo 'Generating data...\n'
# python generateData_dnhp TMax Alpha_Multiplier RandomSeed
python generateData_dnhp.py 100000 1 123132
echo 'Data generated...\n'

echo 'Accumulating top100 parents for each event\n'
g+++ top100Par getStoreTopKCandParents.cpp
# ./top100Par eventsFileName top-100-parents-file top-100-parents-exp-time-diff
./top100Par events_1L.txt top100Parents.txt  top100ParentExps.txt
echo 'Stored top100 parents for each event\n'