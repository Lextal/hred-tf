import re
import numpy as np


def prepare_splitters(tokens):
    return [re.compile(t) for t in tokens]


def get_stats(tokens, line):
    """
    Returns a vector with length same as tokens which contains the maximal number of tokens of every type
    For example, for a sentence with tokens that split it into words, the result will be
        [max_word_length, amount_of_words]
    """

    def max_number_of_elements(token, sequence):
        split = re.split(token, sequence)
        return max(map(len, split)), split

    result = []
    buf = [line]
    for t in reversed(tokens): # starting from the top of hierarchy
        final_max_length = -1
        new_buf = []
        for line in buf:
            max_length, sub_result = max_number_of_elements(t, line)
            final_max_length = max(final_max_length, max_length)
            new_buf.extend(sub_result)
        buf = new_buf
    return np.asarray(reversed(result))


def analyze_file(path, tokens):
    data = open(path).readlines()