from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import reuters
# from nltk.corpus import twitter_samples
from config import config

corpus_words = reuters.words()

with open(config['categories_path'], 'r') as categories_file:
	categories = []
	for line in categories_file:
		label_name, label = line.split(" ")
		label_name = label_name.replace("_", " ")
		label_name = label_name.replace("/", " ")
		categories.append(label_name)


words_not_in_corpus = []
for category_words in categories:
	words = category_words.split(" ")
	# print(words)
	for word in words:
		if word not in corpus_words:
			print(word)
			words_not_in_corpus.append(word)

print(words_not_in_corpus)