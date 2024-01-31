import nltk

from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from nltk.probability import FreqDist, ConditionalFreqDist

# Load the Brown corpus
nltk.download('brown')
nltk.download('universal_tagset')
corpus = brown.tagged_sents(categories='news', tagset='universal')


# Split the corpus into training and test sets
train_size = int(0.9 * len(corpus))
train_set, test_set = corpus[:train_size], corpus[train_size:]


# i. Compute the most likely tag for each word in the training set
word_tag_freq = ConditionalFreqDist((word.lower(), tag) for sentence in train_set for (word, tag) in sentence)
most_likely_tag = {word: freqdist.max() for word, freqdist in word_tag_freq.items()}

# Set the most likely tag for unknown words to "NN"
most_likely_tag.default_factory = lambda: 'NN'

# ii. Compute the error rate for known and unknown words in the test set
total_words = 0
correct_known_words = 0
correct_unknown_words = 0

for sentence in test_set:
    for word, actual_tag in sentence:
        total_words += 1
        predicted_tag = most_likely_tag[word.lower()]

        if actual_tag == predicted_tag:
            if word.lower() in word_tag_freq:
                correct_known_words += 1
            else:
                correct_unknown_words += 1

# Compute error rates
error_rate_known_words = 1 - (correct_known_words / total_words)
error_rate_unknown_words = 1 - (correct_unknown_words / (total_words - len(word_tag_freq)))
total_error_rate = 1 - ((correct_known_words + correct_unknown_words) / total_words)

print(f"Error rate for known words: {error_rate_known_words:.4f}")
print(f"Error rate for unknown words: {error_rate_unknown_words:.4f}")
print(f"Total error rate: {total_error_rate:.4f}")



# # (b) Implementation of the most likely tag baseline
#
# def most_likely_tag_baseline():
#     brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
#
#



