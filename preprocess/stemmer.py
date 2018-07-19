import nltk
from nltk.stem.lancaster import LancasterStemmer


def stem_sentence(sentence):

    """
    
    Takes STRING as an input, returns a STEMMED STRING
    
    """

    ps = nltk.stem.PorterStemmer()
    ls = LancasterStemmer()

    stemmed_sentence = []
    for word in sentence.strip().split(" "):
        #stemmed_sentence.append(ps.stem(word))
        stemmed_sentence.append(ls.stem(word))
    new_sentence = " ".join(stemmed_sentence)
    return new_sentence
