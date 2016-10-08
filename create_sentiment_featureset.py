#Need to create our lexicon - some global lexicon
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter

#word_tokenize separates out words from a sentence
#lemmatizer running, ran, run --> run, tense consolidation
# i liked this product, i like this product, edge case


lemmatizer = WordNetLemmatizer()
hm_lines = 100000000


def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []

	#Remove super common words, but remove infrequent terms
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)

	return l2


def sample_handling(sample, lexicon, classification):
	feature_set = [] # nested lists ]

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower()):
					features[index_value] = 1
			features = list(features)
			feature_set.append([features, classification])

	return feature_set