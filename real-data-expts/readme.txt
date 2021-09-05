# ReadMe for real data expts.

A set of extracted tweets in the file named "cleanedTweets.txt" by the US politics members (mentioned in the paper) is present in the preprocessing folder.
These tweets were extracted using the Twitter API and for each member maximum of 3200 tweets were allowed by the API.
The folder also contains the list of the members along with their IDs.
These set of tweets in "cleanedTweets.txt" are then preprocessed and tokenized using "cleanTweets.py" into two files named "eventsFile" and "dataFile".
Each line in "eventsFile" consists of timestamp <space> user-id <space> parent-id <space> topic-id <space> level-id.
The parent-id and the topic-id here is set to -1 for all the events.
The "dataFile" consists of tokenized set of word-ids for each event on each line.  

We split the dataset first as mentioned in the paper using the probabilities pdata and ptest.
To do that, "splitTrainTestSelEventCandParThreshold.py" consists of variables "eventSelectionProb" and "threshold" that take care of these probabilities respectively.
Set these values to generate split dataset as per the different probabilities.

This will result into a generation of events file with the name as: 
	events_scaledTime_EventSelect_<pdata>_testThresh_<ptest>_set_<setNumber>.txt

As mentioned in the paper for each value of pdata we generate three files and report average values over these three sets of events generated.

Once the Train-Test sets are generated, you are good to run the inference models (DNHP, HMHP, NetHawkes(nhLdamm folder)) on the train data now.
The code for each model is in the respective folder.


Each of the models folder have a "evalTestLL.sh" file that runs the inference model and evaluations after that.
The inference code runs total 300 iterations where BURN-IN period is 200 iterations and samples are taken at every 10 iteration after 200 iteration.
This inference code generates a number of files -- 
(a) estimated user-user influence (userUserInf.txt)
(b) estimated topic-topic influence (topicTopicInf.txt)
(c) parent assignment samples (parentAssignments.txt)
(d) topic assignment samples (topicAssignments.txt)
(e) average distribution over candidate parent assignments (parentAssignmentAvgProbFile.txt)

These files are generated for the Train set data only. I.e. the model learns the required parameters and the latent variables for each event in the Train set.

Using the learned parameters the goal is to estimate the Log-Likelihood of the corresponding Test set.