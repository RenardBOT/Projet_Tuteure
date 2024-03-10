from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# ----- SHANNON ENTROPY
# mesure de la quantité d'information contenue dans un message. 
# interprété dans le cas d'un nom : plus le nom est "complexe" (c'est-à-dire, plus il contient de caractères différents),
# plus l'entropie est élevée

def shannon_entropy(nom):
    if len(nom) == 0:
        return 0
    entropy = 0
    for char in set(nom):
        p_char = nom.count(char) / len(nom)
        entropy += - p_char * log(p_char, 2)
    return entropy

# ----- UPPER / LOWER
# Nombre de lettres majuscules et minuscules dans le nom (normalisé sur 1)

def upper_count(nom):
    count = sum(1 for c in nom if c.isupper())
    return count / len(nom)

def lower_count(nom):
    count = sum(1 for c in nom if c.islower())
    return count / len(nom)


# ----- NGRAM
# un n-gramme est une sous-séquence de n éléments consécutifs d'une séquence donnée.

def ngram(nom, n):
    ngram = []
    if len(nom) < n:
        return [nom]
    for i in range(len(nom) - n + 1):
        ngram.append(nom[i:i+n])
    return ngram

def bigram(nom):
    return ngram(nom, 2)

def trigram(nom):
    return ngram(nom, 3)


# ----- TFIDF
# Term Frequency - Inverse Document Frequency
# mesure statistique utilisée pour évaluer l'importance des n-grammes dans un ensemble de noms

def tfidf(noms):
    list_ngrams = [bigram(nom) for nom in noms]
    strings_ngrams = []
    for ngram in list_ngrams:
       strings_ngrams.append(' '.join(ngram)) 

    # 
    vectorizer = CountVectorizer() 
    tfidftrans = TfidfTransformer() 

    word_freq = vectorizer.fit_transform(strings_ngrams)
    tfidf = tfidftrans.fit_transform(word_freq)
    tfidf_list = tfidf.toarray().tolist()
    
    return tfidf_list




# ----- MAIN

if __name__ == '__main__':
    noms = ['aabbcc','aaaadd']

    for nom in noms:
        print(nom, bigram(nom))
    tf = tfidf(noms)
    print(tf)

    print(shannon_entropy('aAaAaA_1a2A'))
    print(upper_count('aAaAaA_1a2A'))
    print(lower_count('aAaAaA_1a2A'))