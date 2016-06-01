import re
import numpy as np
import tensorflow as tf


def prepare_splitters(tokens):
    return [re.compile(t) for t in tokens]

DEFAULT_VOCAB = [''] + [chr(ord('a') + i) for i in range(26)] + [str(i) for i in range(10)] + [' ']-
DEFAULT_VOCAB = dict(zip(DEFAULT_VOCAB, range(len(DEFAULT_VOCAB))))
DEFAULT_TOKENS = prepare_splitters([' ', '\. '])
FILTER_NON_ENG = re.compile('[^a-z ]+')


def get_stats(line, tokens=DEFAULT_TOKENS):
    """
    Returns a vector with length same as tokens which contains the maximal number of tokens of every type
    For example, for a sentence with tokens that split it into words, the result will be
        [max_word_length, amount_of_words]
    """
    result = []
    buf = re.split(tokens[0], line)
    max_chars = max([len(x) for x in buf])
    buf = [line]
    for t in reversed(tokens):
        new_buf = [re.split(t, x) for x in buf]
        result.append(max([len(x) for x in new_buf]))
        buf = []
        for sublist in new_buf:
            buf.extend(sublist)
    result.append(max_chars)
    return np.asarray(result[::-1])


def analyze_file(path, tokens):
    stats = []
    n = 0
    splitters = prepare_splitters(tokens)
    with tf.gfile.GFile(path) as data:
        for line in data:
            stats.append(get_stats(line.strip(), splitters))
            n += 1
            if n % 100000 == 0:
                print('{} lines processed'.format(n))
        stats = np.asarray(stats)
        result = [np.max(stats[:, i]) for i in range(stats.shape[1])]
        return result


def encode_sent(sent, levels, vocab=DEFAULT_VOCAB):
    word_size, n_words = levels[0] + 1, levels[1]
    result = []
    words = sent.split()
    for _ in range(len(words), n_words):
        words.append('')
    for w in words:
        for c in range(word_size):
            if c == len(w):
                result.append(' ')
            else:
                if c > len(w):
                    result.append('')
                else:
                    result.append(w[c])
    return [vocab[c] for c in result]


def encode_file(source_path, output_path, levels):
    with tf.gfile.GFile(source_path) as source:
        with tf.gfile.GFile(output_path, mode='w') as target:
            s = source.readline()
            while s:
                h = encode_sent(s, levels)
                target.write(' '.join(map(str, h)) + '\n')
                s = source.readline()


def validate(path, levels):
    """
    Validation for two-level hierarchy only
    """
    checksum = (levels[0] + 1) * levels[1]
    with tf.gfile.GFile(path) as f:
        s = f.readline()
        while s:
            if len(s.split()) != checksum:
                print('String had unmatching amount of tokens: {}'.format(len(s.split())))
                print(s.split())
                return False
            s = f.readline()
    print('File {} is valid with fixed sequence length = {}'.format(path, checksum))
    return True


def split_source_and_target(seq_file, data_dir):
    data = open(seq_file).readlines()
    source = open(data_dir + 'source.txt', 'w')
    target = open(data_dir + 'target.txt', 'w')
    for i in range(len(data) - 1):
        source.write(data[i])
        target.write(data[i + 1])