from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = 'This is Generative AI workshop'

sentences = sent_tokenize (text)

words = word_tokenize (text)

for i in words:
    print (i)
