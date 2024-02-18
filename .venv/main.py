import copy

import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, FreqDist
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

BUFFER = "***************"

START_TAG = "sta!rt_tag"
START_WORD = "sta!rt_word"
DEFAULT_WORD = "default"
DEFAULT_TAG = "NN"
PSEUDO_SET = {"twoDigitNum", "fourDigitNum",
              "lowerCase", "UpperCase", "mixedCase",
              "containsDigit", "initCap", "other", "otherNum"}


def divide_data(corpus, train_ratio=0.9):
    train_size = int(train_ratio * len(corpus))
    return corpus[:train_size], corpus[train_size:]


def remove_suffixes(tag):
    """
    remove the "-" or "+" suffixes.
    """
    return tag if tag[0] in ["-", "+", "*"] else tag.split("-")[0].split("+")[0].split("*")[0]


# Count occurrences of tags and tag bigrams in the training set
def build_bigram_and_unigram(train_set):
    """
    :param train_set:
    :return: bigram and unigram dictionaries.
    """
    bigram_fd = FreqDist()
    unigram_fd = FreqDist()
    for sentence in train_set:
        sentence = [(START_WORD, START_TAG)] + sentence
        tags = [tag for word, tag in sentence]
        bigrams_tags = list(nltk.bigrams(tags))
        unigram_fd.update(tags)
        bigram_fd.update(bigrams_tags)
    return bigram_fd, unigram_fd


def build_words_dict(train_set):
    words_dict = {}
    for sentence in train_set:
        for word, tag in sentence:
            words_dict[word] = words_dict.get(word, 0) + 1
    return words_dict


def create_pseudo_word(word):
    if word.isdigit():
        if len(word) == 2:
            pseudo_word = "twoDigitNum"
        elif len(word) == 4:
            pseudo_word = "fourDigitNum"
        else:
            pseudo_word = "otherNum"
    elif word.isalpha():
        if word.islower():
            pseudo_word = "lowerCase"
        elif word.isupper():
            pseudo_word = "UpperCase"
        else:
            pseudo_word = "mixedCase"
    elif word.isalnum():
        pseudo_word = "containsDigit"
    elif word[0].isupper() and word[1:].islower():
        pseudo_word = "initCap"
    else:
        pseudo_word = "other"
    return pseudo_word


def create_pseudo_train_set(train_set):
    word_appearances_dict = build_words_dict(train_set)
    uncommon_words = (word for word, appearances in word_appearances_dict.items() if appearances < 5)
    new_train_set = []
    for sentence in train_set:
        new_sentence = []
        for word, tag in sentence:
            if word in uncommon_words:
                new_sentence.append((create_pseudo_word(word), tag))
            else:
                new_sentence.append((word, tag))
        new_train_set.append(new_sentence)
    return new_train_set


def calculate_transition_probability(unigram_fd: dict, bigram_fd: dict) -> dict:
    """
    Calculate transition probabilities from each tag to every other tag.

    :param unigram_fd: Dictionary containing counts of each tag in the corpus.
    :param bigram_fd: Dictionary containing counts of each bigram (tag1, tag2) in the corpus.
    :return: Dictionary containing transition probabilities from each tag to every other tag.
    """
    # Initialize all possible transitions with zero probability
    transition_probability = {tag1: {tag2: 0.0 for tag2 in unigram_fd.keys()} for tag1 in unigram_fd.keys()}
    for (tag, next_tag) in bigram_fd:
        transition_probability[tag][next_tag] = bigram_fd[(tag, next_tag)] / unigram_fd[tag]
    return transition_probability


def calculate_emission_probability(word_tag_freq_dict: dict, tags_list: dict, add_one=False, pseudo=False):
    """
    :param word_tag_freq_dict: {word : {tag, number of tag appearances with the word}}
    :return: Dictionary of emission probabilities, {word: {tag : probability of the word given the tag}}
    """
    emission_probability = {word: {tag: 0.0 for tag in tags_list} for word in word_tag_freq_dict.keys()}
    if not add_one and not pseudo:
        emission_probability[DEFAULT_WORD] = {tag: 0.0 if tag != DEFAULT_TAG else 1.0 for tag in tags_list}
    tags_appearances_dict = {}
    for freqdist in word_tag_freq_dict.values():
        for tag, freq in freqdist.items():
            tags_appearances_dict[tag] = tags_appearances_dict.get(tag, 0) + freq
    total_words = len(word_tag_freq_dict.keys())
    return ret_probabilties(add_one, emission_probability,
                            tags_appearances_dict, total_words,
                            word_tag_freq_dict)


def ret_probabilties(add_one, emission_probability, tags_appearances_dict, total_words, word_tag_freq_dict):
    for word in word_tag_freq_dict:
        for tag, tag_appearances in word_tag_freq_dict[word].items():
            if add_one:
                emission_probability[word][tag] = (tag_appearances + 1) / (tags_appearances_dict[tag] + total_words)
            else:
                emission_probability[word][tag] = tag_appearances / tags_appearances_dict[tag]
    return emission_probability


# c.ii Implementing the Viterbi algorithm

def calculate_error_rates(test_set, word_tag_freq_dict):
    total_words = 0
    correct_tags = 0
    correct_known_words = 0
    correct_unknown_words = 0
    total_unknown_word = 0
    total_known_words = 0

    for sentence in test_set:
        for word, actual_tag in sentence:
            total_words += 1

            # Use get method to handle unknown words
            predicted_tag = most_likely_tag.get(word, most_likely_tag[DEFAULT_WORD])

            if actual_tag == predicted_tag:
                correct_tags += 1
                if word in word_tag_freq_dict:
                    correct_known_words += 1
                    total_known_words += 1
                else:
                    correct_unknown_words += 1
                    total_unknown_word += 1
            else:
                if word not in word_tag_freq_dict:
                    total_unknown_word += 1
                else:
                    total_known_words += 1

    # Compute error rates
    print_error_rates(correct_known_words, correct_tags, correct_unknown_words,
                      total_known_words, total_words, total_unknown_word, "")


# Use get method to handle unknown words

def find_next_step(pi: List[List[tuple[float, int]]],
                   emission_probability: dict,
                   transition_probability: dict[str, dict[str, float]],
                   cur_w_ind: int,
                   cur_w: str,
                   cur_t: str,
                   tags_list: List[str]) -> tuple[float, int]:
    """
    :param pi: matrix with n columns (where n is the length of the sentence) and m rows (where m is the number of tags).
            each cell consist of tuple of the probability of the most likely word-and-tag, and the index of the tag in
             the tags list
    :param emission_probability: Dictionary of emission probabilities, {word: {tag : probability of the word given the tag}}
    :param transition_probability: Dictionary of transition probabilities, {prev_tag : {tag : probability of the tag given the prev one}}
    :param cur_w_ind:
    :param cur_tag:
    :return:
    """
    m = len(tags_list)  # number of tags, include START_TAG
    p = []
    # for t in range(1, m):
    for t in range(m):
        prev_t: str = tags_list[t]
        if prev_t != START_TAG:
            prob = pi[t][cur_w_ind - 1][0]
            if cur_t not in emission_probability[cur_w]:
                emission_probability[cur_w][cur_t] = 0.0
            emis_p = emission_probability[cur_w][cur_t]
            tran_p = transition_probability[prev_t][cur_t]
            p.append(prob * emis_p * tran_p)

    max_prob = max(p)
    for t in range(m):
        # for t in range(m):
        prev_t = tags_list[t]
        if prev_t != START_TAG:  # Exclude START_TAG from transition probabilities
            cur_best = (pi[t][cur_w_ind - 1][0] *
                        emission_probability[cur_w][cur_t] *
                        transition_probability[prev_t][cur_t])
            if cur_best == max_prob:
                return max_prob, t


def viterbi(sentence: List[str], emission_probability: dict,
            transition_probability, tags_list: List[str]) -> List[str]:
    """
    :param sentence:
    :param emission_probability: Dictionary of emission probabilities, {word: {tag : probability of the word given the tag}}
    :param transition_probability: Dictionary of transition probabilities, {tag : {next_tag : probability of the next_tag given the tag}}
    :param tags_list: List of all possible tags
    :return: most likely sequence of tags by viterbi algorithm
    """
    n = len(sentence)  # include START_WORD
    m = len(tags_list)  # include START_TAG
    pi = [[(0, '') if word != 0 else (1, START_TAG) for word in range(n)] for tag in range(m)]
    for word_ind in range(1, n):
        for tag_ind in range(m):
            cur_word = sentence[word_ind]
            cur_tag = tags_list[tag_ind]
            pi[tag_ind][word_ind] = find_next_step(pi, emission_probability, transition_probability, word_ind,
                                                   cur_word, cur_tag, tags_list)
    last_cell = max(range(m), key=lambda row: pi[row][n - 1][0])
    return find_best_tag_sequence(pi, last_cell, tags_list)


def find_best_tag_sequence(pi: List[List[tuple[float, str]]], last_cell: int, tags_list: List[str]) \
        -> List[str]:
    """
    Finds the best tagged sequence for a sentence.
    The function retrieve this sequence from the dynamic programing matrix by backtracking on the saved cells.
    :param pi: matrix with n columns (where n is the length of the sentence) and m rows (where m is the number of tags).
             pi[i][j] = (float, int) = (probability of the sequence with i words that ends with the tag tags_list[j],
              and the index - in tags_list -  of the previous tag that maximizes the probability of the sequence)
    :param last_cell: index of the most likely previous tag in the tags list (number between 1 to m)
    :param tags_list: list[str], list of all the available tags
    :return:
    """
    n = len(pi[0])  # number of words in the sentence  include START_WORD
    tags_sequence = [""] * (n - 1)  # include START_WORD
    for col in range(n - 1, 0, -1):  # the tags of the previous word in the sentence
        chosen_tag = last_cell
        tags_sequence[col - 1] = tags_list[chosen_tag]
        last_cell = pi[chosen_tag][col][1]
    return tags_sequence


def print_viterbi_error_on_test_set(test_set, unigram_fd, emission_probability, transition_probability,
                                    total_words, tags_appearances, add_one=False, pseudo=False):
    total_predicted_tags = correct_unknown_words = correct_known_words = 0
    total_unknown_words = total_known_words = correct_tags = 0
    tags_list = list(unigram_fd.keys())  # include START_TAG
    unknown_words = set()
    for line in test_set:
        for i in range(len(line)):
            if line[i][0] not in emission_probability.keys():
                total_unknown_words += 1
                if pseudo:
                    line[i] = (create_pseudo_word(line[i][0]), line[i][1])
                elif add_one:
                    emission_probability[line[i][0]] = {line[i][1]: 1 / total_words + tags_appearances[line[i][1]]}
                    unknown_words.add(line[i][0])
                elif not add_one and not pseudo:
                    line[i] = (DEFAULT_WORD, line[i][1])
            else:
                if line[i][0] in unknown_words:
                    total_unknown_words += 1
                else:
                    total_known_words += 1

            if line[i][1] not in tags_list:
                # Handle the case where the tag is not in the list of possible tags
                pass

        sentence = [START_WORD] + [word[0] for word in line]  # include START_WORD
        predicted_tags = viterbi(sentence, emission_probability, transition_probability, tags_list)
        for i in range(len(predicted_tags)):
            total_predicted_tags += 1
            if predicted_tags[i] == line[i][1]:  # if it's the same tag
                correct_tags += 1
                if (line[i][0] in PSEUDO_SET) or (line[i][0] == DEFAULT_WORD) or (line[i][0] in unknown_words):
                    correct_unknown_words += 1
                else:
                    correct_known_words += 1
    print_error_rates(correct_known_words, correct_tags, correct_unknown_words,
                      total_known_words, total_predicted_tags, total_unknown_words, " using Viterbi")


def print_error_rates(correct_known_words, correct_tags, correct_unknown_words,
                      total_known_words, total_predicted_tags, total_unknown_words, text):
    error_rate = 1 if total_predicted_tags == 0 else 1 - (correct_tags / total_predicted_tags)
    known_words_rate = 1 if total_known_words == 0 else 1 - (correct_known_words / total_known_words)
    unknown_words_rate = 1 if total_unknown_words == 0 else 1 - (correct_unknown_words / total_unknown_words)

    print(f"Total error rate{text} is: {error_rate:.4f}")
    print(f"Error rate for known words{text} is: {known_words_rate:.4f}")
    print(f"Error rate for unknown words{text} is: {unknown_words_rate:.4f}")


def print_confusion_matrix(pseudo_training_data, test_set, unigram_fd, emission_probability, transition_probability):
    tags_list = list(unigram_fd.keys())  # include START_TAG
    k = len(tags_list)
    confusion_matrix = [[0 for _ in range(k)] for _ in range(k)]  # create the matrix
    for line in test_set:
        for i in range(len(line)):
            # print("word: ", line[i][0], " tag: ", line[i][1])
            if line[i][0] not in emission_probability:
                line[i] = (create_pseudo_word(line[i][0]), line[i][1])
            if line[i][1] not in tags_list:
                line[i] = (line[i][0], DEFAULT_TAG)

        sentence = [START_WORD] + [word[0] for word in line]
        predicted_tags = viterbi(sentence, emission_probability, transition_probability, tags_list)

        for i in range(len(predicted_tags)):
            true_tag_index = tags_list.index(line[i][1])
            predicted_tag_index = tags_list.index(predicted_tags[i])
            # Exclude START_TAG
            if tags_list[true_tag_index] != START_TAG and tags_list[predicted_tag_index] != START_TAG:
                confusion_matrix[true_tag_index][predicted_tag_index] += 1  # increment corresponding cell

    print_top_error_tags(confusion_matrix, tags_list)

    # Exclude START_TAG from the tags list and confusion matrix
    filtered_tags_list = [tag for tag in tags_list if tag != START_TAG]
    filtered_confusion_matrix = [[confusion_matrix[i][j] for j in range(k) if tags_list[j] != START_TAG] for i in
                                 range(k) if tags_list[i] != START_TAG]

    print_top_error_tags(filtered_confusion_matrix, tags_list)
    plot_confusion_matrix(filtered_confusion_matrix, filtered_tags_list)


def print_top_error_tags(confusion_matrix, tags_list):
    error_counts = []

    # Iterate over the confusion matrix to collect non-zero error counts excluding diagonal cells
    for true_tag_index in range(len(confusion_matrix)):
        for predicted_tag_index in range(len(confusion_matrix[0])):
            if (true_tag_index != predicted_tag_index and
                    tags_list[predicted_tag_index] != START_TAG):  # Exclude diagonal cells and START_TAG predictio
                count = confusion_matrix[true_tag_index][predicted_tag_index]
                if count > 0:
                    true_tag = tags_list[true_tag_index]
                    predicted_tag = tags_list[predicted_tag_index]
                    error_counts.append((true_tag, predicted_tag, count))

    # Sort error counts based on count in descending order
    sorted_error_counts = sorted(error_counts, key=lambda x: x[2], reverse=True)

    # Print the top 20 tags with the highest error counts
    print("Top 20 tags with the highest error counts:")
    for true_tag, predicted_tag, count in sorted_error_counts[:20]:
        print(f"For true tag '{true_tag}', there are '{count}' mistakes with the predicted tag '{predicted_tag}'")


def plot_confusion_matrix(confusion_matrix, tags_list):
    fig = go.Figure(data=go.Heatmap(z=confusion_matrix, x=tags_list, y=tags_list, colorscale='Blues'))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Tag',
        yaxis_title='True Tag',
        xaxis=dict(tickangle=-45),
        width=1300,
        height=800
    )
    fig.show()


if __name__ == "__main__":
    # a.
    # Initialization - Load the Brown corpus

    nltk.download('brown')
    corpus = brown.tagged_sents(categories='news')
    corpus = [[(word, remove_suffixes(tag)) for word, tag in sentence] for sentence in corpus]

    # Split the data
    train_set, test_set = divide_data(corpus)

    # b.i. Compute the most likely tag for each word in the training set
    # {word : {tag : appearances}}
    word_tag_freq_dict = ConditionalFreqDist((word, tag) for sentence in train_set for (word, tag) in sentence)
    total_words = len(word_tag_freq_dict.keys())
    tags_appearances = {}
    for freqdist in word_tag_freq_dict.values():
        for tag, freq in freqdist.items():
            if tag not in tags_appearances:
                tags_appearances[tag] = freq
            else:
                tags_appearances[tag] += freq
    bigram_fd, unigram_fd = build_bigram_and_unigram(train_set)
    transition_probability = calculate_transition_probability(unigram_fd, bigram_fd)
    most_likely_tag = {word: freqdist.max() for word, freqdist in word_tag_freq_dict.items()}

    # Set the most likely tag for unknown words to "NN"
    most_likely_tag.setdefault(DEFAULT_WORD, DEFAULT_TAG)

    # b.ii. Compute the error rate for known and unknown words in the test set
    print("b.ii.")
    calculate_error_rates(test_set, word_tag_freq_dict)
    print(BUFFER)
    # c.
    print("c.iii - regular")

    emission_probability = calculate_emission_probability(word_tag_freq_dict, unigram_fd)
    new_test_set = copy.deepcopy(test_set)
    print_viterbi_error_on_test_set(new_test_set, unigram_fd, emission_probability, transition_probability, total_words,
                                    tags_appearances)

    # d. i
    print(BUFFER)
    print("d.i - add_one")
    new_test_set = copy.deepcopy(test_set)
    add_one_emission_probability = calculate_emission_probability(word_tag_freq_dict, unigram_fd, True)
    print_viterbi_error_on_test_set(new_test_set, unigram_fd, add_one_emission_probability, transition_probability,
                                    total_words, tags_appearances, True, False)

    print(BUFFER)
    print("e.i - pseudo")

    pseudo_train_set = create_pseudo_train_set(train_set)
    new_test_set = copy.deepcopy(test_set)
    pseudo_bigram, pseudo_unigram = build_bigram_and_unigram(pseudo_train_set)
    pseudo_transition_probability = calculate_transition_probability(pseudo_unigram, pseudo_bigram)
    pseudo_word_tag_freq_dict = ConditionalFreqDist(
        (word, tag) for sentence in pseudo_train_set for (word, tag) in sentence)
    for word in PSEUDO_SET:
        if word not in pseudo_word_tag_freq_dict.keys():
            pseudo_word_tag_freq_dict[word] = {DEFAULT_TAG: 1.0}

    pseudo_emission_probability = calculate_emission_probability(pseudo_word_tag_freq_dict, pseudo_unigram, False, True)
    print_viterbi_error_on_test_set(new_test_set, pseudo_unigram, pseudo_emission_probability,
                                    pseudo_transition_probability, total_words, tags_appearances, False, True)

    print(BUFFER)
    print("e.ii - add_one and pseudo")

    new_test_set = copy.deepcopy(test_set)
    ps_add_one_emission_probability = calculate_emission_probability(pseudo_word_tag_freq_dict, pseudo_unigram, True)
    print_viterbi_error_on_test_set(new_test_set, pseudo_unigram, ps_add_one_emission_probability,
                                    pseudo_transition_probability, total_words, tags_appearances, True, True)

    print(BUFFER)
    print("e.iii")

    new_test_set = copy.deepcopy(test_set)
    ps_add_one_emission_probability = calculate_emission_probability(pseudo_word_tag_freq_dict, pseudo_unigram, True)
    print_confusion_matrix(pseudo_train_set, new_test_set, pseudo_unigram, ps_add_one_emission_probability,
                           transition_probability)
