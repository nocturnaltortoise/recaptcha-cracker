from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import reuters
# from nltk.corpus import twitter_samples
from config import config

reuters_words = reuters.words()
gutenberg_words = gutenberg.words()
webtext_words = webtext.words()
brown_words = brown.words()

with open(config['categories_path'], 'r') as categories_file:
	categories = []
	for line in categories_file:
		label_name, label = line.split(" ")
		label_name = label_name.replace("_", " ")
		label_name = label_name.replace("/", " ")
		categories.append(label_name)

# print(words)
for corpus_words in [reuters_words, gutenberg_words, webtext_words, brown_words]:
	words_not_in_corpus = []
	for category_words in categories:
		words = category_words.split(" ")
		for word in words:
			if word not in corpus_words:
				words_not_in_corpus.append(word)
	print(words_not_in_corpus)
