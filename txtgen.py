#! /usr/bin/env python2


from __future__ import division

import os
import re
import numpy
import sys

from collections import defaultdict, OrderedDict
from nltk.tokenize import PunktWordTokenizer


# Set number of grams here.
NGRAMS = 3


def main():

    if NGRAMS <= 1:
        print "Sorry, NGRAMS must be >= 2."
        return

    # If we've already built some models before,
    # rename the file to keep them separate.
    if os.path.isfile(str(NGRAMS) + 'gram_model.txt'):
        os.rename(str(NGRAMS) + 'gram_model.txt', str(NGRAMS) + 'gram_model.old')

    # Make sure training as test files are named correctly!
    training_data_en = 'scifi.txt'

    # Train the ngram models
    print "Building English model."
    en_model = build_model(training_data_en, model_name='en_model')

    gen_random_output(en_model, n=140)

# ------- BUILD MODEL -------------------------------------
def build_model(training_data_file, model_name=None):
    '''
    # Builds a ngram probability model for
    # for training_data_file and returns that
    # model as a dictionary.
    '''

    # Read in the file.
    print "Counting"
    data_file_str = read_file(training_data_file)

    # Remove unwanted characters.
    processed_data_str = preprocess_line(data_file_str)

    # Count ngrams.
    ngram_counts = count_word_ngrams(NGRAMS, processed_data_str)

    # Discount using Good-Turing discounting.
    print "Discounting"
    n_discounts = gt_discount(ngram_counts)

    # Trigram_probs is a defaultdict with ngrams as keys
    # and their probabilities as values.
    print "Probabilizing"
    ngram_probs = estimate_probs(n_discounts)

    return ngram_probs


def read_file(text_file):
    '''
    # Opens and reads text_file. Returns
    # file contents as a string.
    '''

    with open(text_file, 'r') as f:
        file_string = f.read()

    return file_string


def preprocess_line(file_string):
    '''
    # Reads in file string returned by read_file()
    # and removes all characters that are not
    # whitespace, [a-z][A-Z], comma, or period.
    # Changes all characters to lowercase and
    # converts numerals to 0. Replace whitespace
    # with an underscore for ease of viewing.
    '''

    # Convert to lowercase.
    processed_string = file_string.lower()

    # Delete any characters that are not a digit,
    # whitespace, a-z, comma, or period.
#    processed_string = re.sub(r'[^\d\sa-z,.]', r'', processed_string)

    # Convert all digits to 0.
    processed_string = re.sub(r'\d', r'0', processed_string)

    # Replace all whitespace with single space.
    processed_string = re.sub(r'\s', r' ', processed_string)
    
    return processed_string


def count_word_ngrams(n, processed_string):
    '''
    # Counts all word ngrams in processed_string
    # and creates a dictionary of those ngram counts
    # called ngram_counts_dict.
    '''
    pwt = PunktWordTokenizer()
    processed_string = pwt.tokenize(processed_string)

    # This empty dictionary will be filled with
    # the ngrams in processed_string and their frequencies
    ngram_counts_dict = defaultdict(int)

    i = 0
    j = i + n
    for i,_ in enumerate(processed_string):
        ngram = ' '.join(processed_string[i:j])
        i += 1
        j = i + n
        ngram_counts_dict[ngram] = 1

    return ngram_counts_dict


def gt_discount(n_counts):
    '''
    # Good-Turing discounter.
    # Calculates Good-Turing probability of
    # zero-count ngrams, and disocunts
    # all counts in n_counts accordingly.
    # Recasts n_counts as a defaultdict with
    # zero_count_probs as the default value.
    '''

    # Calculate the probability for ngrams with zero count.
    # Equation: P_gt_0 = N_1 / N_total
    N_1 = len([i for i in n_counts.itervalues() if i == 1])
    N = sum(n_counts.itervalues())
    zero_count_probs = (N_1 / N)

    # Calculate updated counts and update values.
    # Equation: discount_c = (c+1) * (N_c+1 / N_c)
    num_ops = len(n_counts.items())
    for (i,(key,value)) in enumerate(n_counts.iteritems()):
        sys.stderr.write('{0}/{1}\r' .format(i, num_ops))

        # Calculate first numerator (c+1)
        num1 = value + 1
        if num1 not in n_counts.itervalues():
            pass

        else:
            # Calculate second numerator (N_c+1)
            num2 = len([n for n in n_counts.itervalues() if n == num1])

            # Calculate denominator (N_c)
            denom = len([n for n in n_counts.itervalues() if n == value])

            # Calculate the new count and
            # update the count dict with the new count
            n_counts[key] = (num1 * num2) / denom

    # Cast n_counts as a defaultdict with zero_count_probs as the default.
    # Default values used in calc_perplexity.
    new_counts = defaultdict(lambda: zero_count_probs, n_counts)

    return new_counts


def estimate_probs(ngram_counts_dict):
    '''
    # Estimates probabilities of ngrams using
    # ngram_counts_dict and returns a new dictionary
    # with the probabilities.
    '''
    ngram_probs_dict = ngram_counts_dict.copy()

    num_ops = len(ngram_counts_dict.items())
    for (i,(key,value)) in enumerate(ngram_counts_dict.iteritems()):
        sys.stderr.write('{0}/{1}\r' .format(i, num_ops))
        lessgrams = [ngram_counts_dict[k] for k in ngram_counts_dict.keys() \
                     if k.split()[:-1] == key.split()[:-1]]
        ngram_probs_dict[key] = value / sum(lessgrams)

    return ngram_probs_dict


def calc_perplexity(test_counts_dict, ngram_probs_dict):
    '''
    # Calculates perplexity of contents of file_string
    # according to probabilities in ngram_probs_dict.
    '''

    test_probs = []

    for ngram, count in test_counts_dict.items():

        # If the ngram doesn't appear in our model, just skip it.
        for n in range(count):

            # Since ngram_probs_dict is a defaultdict as created
            # in gt_discount, it will return the zero count probability
            # if ngram not in ngram_probs_dict.
            logprob = numpy.log2(ngram_probs_dict[ngram])
            test_probs.append(logprob)

    logprob = sum(test_probs)

    norm = logprob / len(test_probs)

    perplexity = numpy.power(2, -norm)

    return perplexity


def gen_random_output(ngram_probs_dict, n=300):
    '''
    # Generate n characters of output randomly selected
    # from the keys of ngram_probs_dict.
    # OrderedDict is used to ensure that each key is
    # paired with its associated value.
    '''

    # Choose a trigram to start with
    string = numpy.random.choice(ngram_probs_dict.keys())

    # Find all trigrams and their probabilities that start 
    # with the last two characters of random_string.
    for i in range(n):
        od = OrderedDict()
        for key,value in ngram_probs_dict.iteritems():
            if key.split()[:2] == string.split()[-2:]:
                od[key] = value

        # Choose one of those trigrams according to the probability
        # distribution we just built.
        try:
            next_tri = numpy.random.choice(od.keys(), p=od.values())
        except:
            return string

        # Get the last character of that trigram and append it
        # to random_string.
        string = ' '.join([string, next_tri.split()[-1]])
    
    return string


if __name__ == '__main__':
    main()
