def get_vocab_list():
    with open('vocab.txt') as f:
        vocablist = []
        for line in f:
            idx, w = line.split()
            vocablist.append(w)

    return vocablist
