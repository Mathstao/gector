import logging
from symspellpy import SymSpell, Verbosity


class SymSpellChecker(obejct):

    def __init__(self):
        self.checker = SymSpell(max_dictionary_edit_distance=2)
        self.checker.load_dictionary('/home/citao/github/symspellpy/frequency_dictionary_en_82_765.txt', 0, 1)
        self.checker.load_bigram_dictionary('/home/citao/github/symspellpy/frequency_bigramdictionary_en_243_342.txt', 0, 2)

    def correct(self, word):
        suggestions = self.checker.lookup(s, Verbosity.CLOSEST, max_edit_distance=2)
        for suggestion in suggestions:
            cor_word = suggestion.term
            logging.info('Spell check: [{}] -> [{}]'.format(word, cor_word))
            return cor_word
        return word

