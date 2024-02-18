import unittest
import nltk
from main import *


class TestHMM(unittest.TestCase):
    """
    Test suite for Hidden Markov Model (HMM) functions.
    """

    def test_divide_data(self):
        """
        Test divide_data function.
        """
        # Test divide_data with a sample corpus
        corpus = ["This is sentence 1.", "And this is sentence 2."]
        train_set, test_set = divide_data(corpus, train_ratio=0.5)
        self.assertEqual(len(train_set), 1)  # 50% of the corpus
        self.assertEqual(len(test_set), 1)  # 50% of the corpus

    def test_divide_data_advanced(self):
        """
        Advanced test cases for divide_data function.
        """
        # Test with an empty corpus
        corpus = []
        train_set, test_set = divide_data(corpus, train_ratio=0.5)
        self.assertEqual(len(train_set), 0)
        self.assertEqual(len(test_set), 0)

        # Test with a single sentence corpus
        corpus = ["This is a single sentence."]
        train_set, test_set = divide_data(corpus, train_ratio=0.5)
        self.assertEqual(len(train_set), 1)
        self.assertEqual(len(test_set), 0)

    def test_calculate_transition_probability(self):
        """
        Test calculate_transition_probability function.
        """
        # Test calculate_transition_probability with sample frequency distributions
        unigram_fd = {'NOUN': 10, 'VERB': 8, 'ADJ': 5}
        bigram_fd = {('NOUN', 'VERB'): 6, ('VERB', 'ADJ'): 4}
        transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
        self.assertAlmostEqual(transition_prob['NOUN']['VERB'], 0.6)
        self.assertAlmostEqual(transition_prob['VERB']['ADJ'], 0.5)

    def test_calculate_transition_probability_advanced(self):
        """
        Advanced test cases for calculate_transition_probability function.
        """
        # Test with empty frequency distributions
        unigram_fd = {}
        bigram_fd = {}
        transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
        self.assertEqual(transition_prob, {})

        # Test with frequency distributions containing only one tag
        unigram_fd = {'NOUN': 10}
        bigram_fd = {}
        transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
        self.assertEqual(transition_prob, {'NOUN': {'NOUN': 0.0}})

        # Test with frequency distributions containing one tag with no transitions to others
        unigram_fd = {'NOUN': 10}
        bigram_fd = {('NOUN', 'NOUN'): 0}
        transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
        self.assertEqual(transition_prob, {'NOUN': {'NOUN': 0.0}})

    def test_calculate_emission_probability(self):
        """
        Test calculate_emission_probability function.
        """
        # Test calculate_emission_probability with sample word-tag frequency dictionaries
        word_tag_freq_dict = {
            'dog': {'NOUN': 5, 'VERB': 1},
            'run': {'NOUN': 1, 'VERB': 5}
        }
        tags_list = ['NOUN', 'VERB']
        emission_prob = calculate_emission_probability(word_tag_freq_dict, tags_list)
        self.assertAlmostEqual(emission_prob['dog']['NOUN'], 5 / 6)
        self.assertAlmostEqual(emission_prob['run']['VERB'], 5 / 6)

    def test_calculate_emission_probability_advanced(self):
        """
        Advanced test cases for calculate_emission_probability function.
        """
        # Test with empty word-tag frequency dictionary
        word_tag_freq_dict = {}
        tags_list = []
        emission_prob = calculate_emission_probability(word_tag_freq_dict, tags_list)
        self.assertEqual(emission_prob, {})

        # Test with tags_list containing only one tag
        word_tag_freq_dict = {'dog': {'NOUN': 5}}
        tags_list = ['NOUN']
        emission_prob = calculate_emission_probability(word_tag_freq_dict, tags_list)
        self.assertEqual(emission_prob, {'dog': {'NOUN': 1.0}})

    def test_build_bigram_and_unigram(self):
        """
        Test build_bigram_and_unigram function.
        """
        # Test with a small training set
        train_set = [[('The', 'DET'), ('cat', 'NOUN'), ('is', 'VERB'), ('running', 'VERB')],
                     [('The', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB')]]
        bigram_fd, unigram_fd = build_bigram_and_unigram(train_set)
        self.assertEqual(unigram_fd['NOUN'], 2)
        self.assertEqual(bigram_fd[('NOUN', 'VERB')], 2)

        # Test with an empty training set
        train_set = []
        bigram_fd, unigram_fd = build_bigram_and_unigram(train_set)
        self.assertEqual(unigram_fd, {})
        self.assertEqual(bigram_fd, {})

    # def test_calculate_transition_probability_edge_cases(self):
    #     """
    #     Edge cases for calculate_transition_probability function.
    #     """
    #     # Test with an empty unigram frequency distribution
    #     unigram_fd = {}
    #     bigram_fd = {('NOUN', 'VERB'): 5, ('VERB', 'NOUN'): 3}
    #     transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
    #     self.assertEqual(transition_prob, {})
    #
    #     # Test with an empty bigram frequency distribution
    #     unigram_fd = {'NOUN': 5, 'VERB': 3}
    #     bigram_fd = {}
    #     transition_prob = calculate_transition_probability(unigram_fd, bigram_fd)
    #     self.assertEqual(transition_prob, {'NOUN': {'NOUN': 0.0, 'VERB': 0.0}, 'VERB': {'NOUN': 0.0, 'VERB': 0.0}})

    # def test_calculate_emission_probability_edge_cases(self):
    #     """
    #     Edge cases for calculate_emission_probability function.
    #     """
    #     # Test with empty word-tag frequency dictionary
    #     word_tag_freq_dict = {}
    #     tags_list = ['NOUN', 'VERB']
    #     emission_prob = calculate_emission_probability(word_tag_freq_dict, tags_list)
    #     self.assertEqual(emission_prob, {})
    #
    #     # Test with empty tags list
    #     word_tag_freq_dict = {'dog': {'NOUN': 5, 'VERB': 3}}
    #     tags_list = []
    #     emission_prob = calculate_emission_probability(word_tag_freq_dict, tags_list)
    #     self.assertEqual(emission_prob, {'dog': {}})

    def test_viterbi(self):
        """
        Test viterbi function.
        """
        # Test viterbi with sample inputs
        sentence = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        emission_prob = {
            'The': {'DET': 1.0},
            'quick': {'ADJ': 1.0},
            'brown': {'ADJ': 1.0},
            'fox': {'NOUN': 1.0},
            'jumps': {'VERB': 1.0},
            'over': {'ADP': 1.0},
            'the': {'DET': 1.0},
            'lazy': {'ADJ': 1.0},
            'dog': {'NOUN': 1.0}
        }
        transition_prob = {
            'DET': {'ADJ': 0.5, 'NOUN': 0.5},
            'ADJ': {'NOUN': 0.5, 'VERB': 0.5},
            'NOUN': {'VERB': 0.5, 'ADP': 0.5},
            'VERB': {'ADP': 0.5, 'DET': 0.5},
            'ADP': {'DET': 0.5, 'ADJ': 0.25, 'NOUN': 0.25}
        }
        tags_list = ['DET', 'ADJ', 'NOUN', 'VERB', 'ADP']
        result = viterbi(sentence, emission_prob, transition_prob, len(tags_list), tags_list)
        self.assertEqual(len(result), len(sentence))

    def test_viterbi_advanced(self):
        """
        Advanced test cases for viterbi function.
        """
        # Test with an empty sentence
        sentence = []
        emission_prob = {}
        transition_prob = {}
        tags_list = []
        result = viterbi(sentence, emission_prob, transition_prob, len(tags_list), tags_list)
        self.assertEqual(result, [])

        # Test with empty emission_probability
        sentence = ['The', 'quick', 'brown', 'fox']
        emission_prob = {}
        transition_prob = {'DET': {'NOUN': 0.5}, 'NOUN': {'VERB': 0.5}}
        tags_list = ['DET', 'NOUN', 'VERB']
        result = viterbi(sentence, emission_prob, transition_prob, len(tags_list), tags_list)
        self.assertEqual(result, [])

        # Test with empty transition_probability
        sentence = ['The', 'quick', 'brown', 'fox']
        emission_prob = {'The': {'DET': 1.0}, 'quick': {'ADJ': 1.0}}
        transition_prob = {}
        tags_list = ['DET', 'NOUN', 'VERB']
        result = viterbi(sentence, emission_prob, transition_prob, len(tags_list), tags_list)
        self.assertEqual(result, [])

        # Test with tags_list containing only one tag
        sentence = ['The', 'quick', 'brown', 'fox']
        emission_prob = {'The': {'DET': 1.0}, 'quick': {'ADJ': 1.0}}
        transition_prob = {'DET': {'NOUN': 0.5}}
        tags_list = ['DET']
        result = viterbi(sentence, emission_prob, transition_prob, len(tags_list), tags_list)
        self.assertEqual(result, [])
