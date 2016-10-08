#Need to create our lexicon - some global lexicon
import nltk
import codecs
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
		with codecs.open(fi, 'r', 'utf-8') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []

	#Remove super common words, but remove infrequent terms
	for w in w_counts:
		if 1000 > w_counts[w] > 20:
			l2.append(w)

	return l2


def sample_handling(sample, lexicon, classification):
	feature_set = [] # nested lists ]

	with codecs.open(sample, 'r', 'utf-8') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			# Can we use CountVectorizer instead?
			"""
				You can create the 
			"""
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] = 1
			features = list(features)
			feature_set.append([features, classification])

	return feature_set


def create_features_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])
	features += sample_handling('neg.txt', lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size*len(features)) #10% 
	"""
	[[features, label], [features, label], [features, label]]
	featuers are like [0,1,0,1,1,0] - so [:,0] is the features
	"""
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y




if __name__ == "__main__":
	print "im running"
	train_x, train_y, test_x, test_y = create_features_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)









