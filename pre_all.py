from spellchecker import SpellChecker
import regex as re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

spell = SpellChecker(language=None)
spell.word_frequency.load_text_file('./word_freq/id_50k.txt')
    
def reduce_repeated(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

pre1 = open(".data/pre1.txt", "a")
data = open(".data/test_data.txt", "r")
for line in data:
    words = line.split(" ")
    for word in words:
        pre1.write(spell.correction(reduce_repeated(word)))
        pre1.write(" ")

#stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stopword removal 
factory2 = StopWordRemoverFactory()
stopword = factory2.create_stop_word_remover()
 
pre2 = open(".data/pre2.txt", "a")
data = open(".data/pre1.txt", "r")
for line2 in data:
  stemmed = stemmer.stem(line2)
  stop = stopword.remove(stemmed)
  pre2.write(stop)