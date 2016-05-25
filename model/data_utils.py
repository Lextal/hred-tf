import re
import numpy as np
import tensorflow as tf

from functools import reduce

def prepare_splitters(tokens):
    return [re.compile(t) for t in tokens]


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


def process_line(line):
    return re.sub(FILTER_NON_ENG, '', line.lower())


def clean_text(source_path, target_path):
    count = 0
    with tf.gfile.GFile(source_path) as source:
        with tf.gfile.GFile(target_path, mode='w') as target:
            s = source.readline()
            while s:
                count += 1
                if count % 100000 == 0:
                    print('{} lines processed'.format(count))
                target.write(process_line(s) + '\n')
                s = source.readline()


def build_hierarchy(tokens, acc):
    if len(tokens) == 0:
        return acc
    else:
        return [build_hierarchy(tokens[1:], re.split(tokens[0], x)) for x in acc]


def encode_file(source_path, output_path, levels):
    word_size, n_words = levels[0], levels[1]
    with tf.gfile.GFile(source_path) as source:
        with tf.gfile.GFile(output_path, mode='w') as target:
            s = source.readline()
            while s:
                h = []
                words = re.split(' ', s.strip())
                for _ in range(len(words), n_words):
                    words.append('')
                for w in range(n_words):
                    for c in words[w]:
                        h.append(ord(c) - ord('a') + 1)
                    for c in range(len(words[w]), word_size):
                        h.append(0)
                target.write(' '.join(map(str, h)) + '\n')
                s = source.readline()


def validate(path, levels):
    checksum = reduce(lambda x, y: x * y, levels)
    with tf.gfile.GFile(path) as f:
        s = f.readline()
        while s:
            if len(s.split()) != checksum:
                print('String had unmatching amount of tokens: {}'.format(len(s.split())))
                print(s.split())
                return False
            s = f.readline()
    return True


def split_source_and_target(seq_file, data_dir):
    data = open(seq_file).readlines()
    source = open(data_dir + 'source.txt', 'w')
    target = open(data_dir + 'target.txt', 'w')
    for i in range(len(data) - 1):
        source.write(data[i])
        target.write(data[i + 1])