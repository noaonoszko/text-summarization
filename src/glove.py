import numpy as np


class Glove:
    def __init__(self, glove_dir):
        self.glove_dir = glove_dir

    def load_glove(self, size):
        path = self.glove_dir + "glove.6B." + str(size) + "d.txt"
        wordvecs = {}
        counter = 0
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                tokens = line.split(" ")
                vec = np.array(tokens[1:], dtype=np.float32)
                wordvecs[tokens[0]] = vec
                if counter % 100000 == 0:
                    print("load_glove: parsing line ", counter)
                counter += 1

        return wordvecs

    def fill_with_gloves(self, word_to_id, emb_size, wordvecs=None):
        if not wordvecs:
            wordvecs = load_glove(emb_size)

        n_words = len(word_to_id)
        res = np.zeros([n_words, emb_size], dtype=np.float32)
        n_not_found = 0
        for word, id in word_to_id.iteritems():
            if word in wordvecs:
                res[id, :] = wordvecs[word]
            else:
                n_not_found += 1
                res[id, :] = np.random.normal(0.0, 0.1, emb_size)
        print("n words not found in glove word vectors: " + str(n_not_found))

        return res
