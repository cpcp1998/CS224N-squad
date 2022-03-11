from collections import Counter, defaultdict
import re
import tqdm
import json


class Tokenizer:
    def __init__(self, merges, chars):
        self.raw_chars = chars
        self.raw_merges = merges
        self.merges = [(re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)"), "".join(pair)) for pair in merges]
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "<w>": 2}
        for c in chars: self.word2idx[c] = len(self.word2idx)
        for p in merges: self.word2idx["".join(p)] = len(self.word2idx)

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def train_bpe(words, count):
        # words_lower = defaultdict(int)
        # for k, v in words.items():
        #     words_lower[k.lower()] += v
        # words = words_lower
        words = {"<w> " + " ".join(k): v for k, v in words.items()}
        merges = []
        freq = None
        for _ in tqdm.trange(count):
            freq = defaultdict(int)
            for word, c in words.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    freq[(symbols[i], symbols[i+1])] += c

            pair = max(freq, key=lambda x: freq[x])
            merges.append(pair)

            p = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
            words = {p.sub("".join(pair), k): v for k, v in words.items()}
        if freq: print("least frequent merge:", max(freq.values()))
        return merges

    def tokenize(self, word):
        word = "<w> " + " ".join(word)
        for p, repl in self.merges:
            word = p.sub(repl, word)
        subword = word.split()
        idx = [self.word2idx.get(s, 1) for s in subword]
        return idx

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.raw_merges, "chars": self.raw_chars}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["merges"], data["chars"])
