import logging
from symspellpy import SymSpell, Verbosity
from hunspell import Hunspell

class SymSpellChecker(object):

    def __init__(self):
        self.checker = SymSpell(max_dictionary_edit_distance=2)
        self.checker.load_dictionary('/home/citao/github/symspellpy/frequency_dictionary_en_82_765.txt', 0, 1)
        self.checker.load_bigram_dictionary('/home/citao/github/symspellpy/frequency_bigramdictionary_en_243_342.txt', 0, 2)

    def correct(self, word):
        suggestions = self.checker.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        for suggestion in suggestions:
            cor_word = suggestion.term
            logging.info('Spell check: [{}] -> [{}]'.format(word, cor_word))
            return cor_word
        return word

    def correct_text(self, text):
        cor_list = []
        for word in text.split(' '):
            suggestions = self.checker.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            cor_flag = False
            for suggestion in suggestions:
                cor_word = suggestion.term
                cor_list.append(cor_word)
                cor_flag = True
                break
            if not cor_flag:
                cor_list.append(word)
        return ' '.join(cor_list)


class HunspellChecker(object):

    def __init__(self):
        self.checker = Hunspell()

    def correct(self, word):
        if self.checker.spell(word) == True:
            return word
        else:
            res = self.checker.suggest(word)
            if res:
                return res[0]
            else:
                return word
