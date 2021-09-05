# -*- coding: utf-8 -*-

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import preprocessor as p
import sys
import re
# the above is for tweetProcessor package for twitter tweets...

# p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)

stop_words = set(stopwords.words('english'))
stop_words.add("RT")
stop_words.add("rt")
stop_words.add("I'm")
stop_words.add("&amp")

# print stop_words

# '''

printable = set(string.printable)

puncs = set(string.punctuation)

websiteWords = ['www.', '.com', '.gov', '.net', '.org', 'http:', 'https:']

# for ch in ['\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'','!','^','&', ';']:

fPtr = open("combinedTweetsWithEventIdUserId_sorted.txt", "r")
# fPtr = ["132_866,132,876925019,1217826143, retweetin@Gillize:PLz Don't U.S. "]
# I will be 999 1999 12,000 1pm attending the State http://www.abc.org/tcot of the Union where President Bush #tcot, “no new ideas.”"]

writeToFptr = open("cleanedTweets.txt", "w")

linenum = 0

for line in fPtr:

	flds = line.strip().split(',')
	# print flds
	uidTweetNum = flds[0].strip()
	uid = int(flds[1].strip())
	tweetId = int(flds[2].strip())
	tstamp = int(flds[3].strip())

	tweetText = ' '.join([s.strip() for s in flds[4:]])

	tokens = tweetText.split()

	cleanedTokens = []

	for eachToken in tokens:
		flag = 1
		# lowerToken = eachToken.lower()
		lowerToken = eachToken
		lowerToken = lowerToken.strip()
		# print lowerToken

		# remove mentions to users...
		if lowerToken.find("@") > -1:
			flag = 0
			continue

		lowerToken = p.clean(lowerToken)
		# print lowerToken
		lowerToken = lowerToken.encode('ascii', 'replace') 

		# print lowerToken
		# check for number, which might be separated with a comma
		# lowerToken = lowerToken.replace(",", "")

		# replace the numbers/digits with empty string...
		lowerToken = re.sub("\d", "", lowerToken)
		
		# remove apostrophe s
		lowerToken = lowerToken.replace("'s", "") 			# remove apostrophe s
		lowerToken = lowerToken.replace("?s", "") 			# remove apostrophe s when replaced because of ignored ascii characters

		# print lowerToken
		# remove website related words
		for webWord in websiteWords:
			if lowerToken.find(webWord) > -1:
				# print "hi"
				flag = 0
				break


		if lowerToken == "U.S." or lowerToken == "U.S":
			lowerToken = "US"

		if flag == 1:

			# remove some special characters...
			for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!','$','!','^','&', ';', '/', '=', '?', '%', '"', ':', 'amp', '&']: # , '\'']:
				lowerToken = lowerToken.replace(ch, " ")
				# print ch, lowerToken

			lowerToken = lowerToken.replace("'", "")
			# print lowerToken

			setLowerTokens = lowerToken.split(" ")

			for eachLowerToken in setLowerTokens:

				eachLowerToken = eachLowerToken.strip()

				# remove words like am pm
				if eachLowerToken == "am" or eachLowerToken == "pm" or eachLowerToken.lower() == "rt":
					continue

				# if eachLowerToken.lower() not in stop_words:
				# 	if len(eachLowerToken) > 0:
				# 		cleanedTokens.append(eachLowerToken)
				if len(eachLowerToken) > 0:
					cleanedTokens.append(eachLowerToken)


	# print linenum, cleanedTokens

	writeStr = str(uidTweetNum) + ";" + str(uid) + ";" + str(tstamp) + ";" + " ".join([x for x in cleanedTokens]) + "\n"
	writeToFptr.write(writeStr)

	linenum += 1


fPtr.close()
writeToFptr.close()