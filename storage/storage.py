from collections import defaultdict

import random


class Storage(object):
    def __init__(self):
        self.words2roots = dict()
        self.roots2words = defaultdict(set)
        self.all_words = set()

    def add_word(self, word, root):
        self.words2roots[word] = root
        self.roots2words[root].add(word)
        self.all_words.add(word)

    def if_cognate(self, word_1, word_2):
        return self.words2roots[word_1] == self.words2roots[word_2]

    def random_sample(self, size, cognate_percent):
        sample = list()
        for i in range(size * cognate_percent // 100):
            random_root = random.choice(list(self.roots2words.keys()))
            sample.append((random.choice(list(self.roots2words[random_root])),
                           random.choice(list(self.roots2words[random_root])),
                           True))

        while len(sample) < size:
            random_root_1, random_root_2 = random.choice(list(self.roots2words.keys())), random.choice(list(self.roots2words.keys()))
            if random_root_1 == random_root_2:
                continue
            sample.append((random.choice(list(self.roots2words[random_root_1])),
                           random.choice(list(self.roots2words[random_root_2])),
                           False))

        random.shuffle(sample)
        return sample

    def really_random_sample(self, size):
        sample = list()
        for i in range(size):
            word_1 = random.choice(list(self.all_words))
            word_2 = random.choice(list(self.all_words))
            sample.append((word_1,
                           word_2,
                           self.if_cognate()))
