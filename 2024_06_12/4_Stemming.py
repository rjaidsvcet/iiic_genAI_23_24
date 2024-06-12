from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

porter = PorterStemmer()
snow = SnowballStemmer(language = 'english')
lancaster = LancasterStemmer()

words = ['play', 'plays', 'played', 'playing', 'player']

# stemmed = list()
# for w in words:
#     stemmed_words = lancaster.stem(w)
#     stemmed.append(stemmed_words)

stemmed = [porter.stem(x) for x in words]

print(stemmed)