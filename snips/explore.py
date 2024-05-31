chars = set()
labels = set()

def add_to_vocab(f):
    global chars, labels
    for line in f.readlines():
        tokens = line.split()
        for token in tokens[:-2]:
            word, label = token.split(":")
            labels.add(label)
            for char in word:
                if char == '.':
                    print(word, token)
                chars.add(char)

add_to_vocab(open("data/train_dev"))
add_to_vocab(open("data/test"))

print(sorted(list(chars)))
print(sorted(list(labels)))