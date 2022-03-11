from collections import Counter, defaultdict
import spacy
import setup
from subword import Tokenizer


def main():
    setup.nlp = spacy.blank("en")
    word_counter = Counter()
    char_counter = Counter()
    dataset = setup.process_file("data/train-v2.0.json", "train", word_counter, char_counter)
    total = sum(char_counter.values())
    cumulative = 0
    chars = []
    for char, count in char_counter.most_common():
        if cumulative > total * 0.999:
            break
        chars.append(char)
        cumulative += count
    print("char count:", len(chars), chars)
    print("before bpe:", sum(len(k) * v for k, v in word_counter.items()))
    merges = Tokenizer.train_bpe(word_counter, 0)
    tokenizer = Tokenizer(merges, chars)
    total_len = 0
    unk_count = 0
    for k, v in word_counter.items():
        idx = tokenizer.tokenize(k)
        total_len += len(idx) * v
        unk_count += sum(i == 1 for i in idx) * v
    print("after bpe:", total_len, unk_count / total_len)
    tokenizer.save("data/bpe.json")

if __name__ == "__main__":
    main()
