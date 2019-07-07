import numpy as np
from getVocabList import get_vocab_list
from nltk.stem.porter import PorterStemmer
import re


def process_email(email_contents):
    vocablist = get_vocab_list()
    word_indices = []

    # ===================== Preprocess Email =====================

    email_contents = email_contents.lower()

    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Any numbers get replaced with the string 'number'
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # The '$' sign gets replaced with 'dollar'
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================
    print('\n==== Processed Email ====\n\n')

    l = 0

    stemmer = PorterStemmer()

    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)

        if len(token) < 1:
            continue

        # ===================== Your Code Here =====================
        # Instructions : Fill in this function to add the index of token to
        #                word_indices if it is in the vocabulary. At this point
        #                of the code, you have a stemmed word frome email in
        #                the variable token. You should look up token in the
        #                vocab_list. If a match exists, you should add the
        #                index of the word to the word_indices nparray.
        #                Concretely, if token == 'action', then you should
        #                look up the vocabulary list the find where in vocab_list
        #                'action' appears. For example, if vocab_list[18] == 'action'
        #                then you should add 18 to the word_indices array.

        for i in range(len(vocablist)):
            if vocablist[i] == token:
                word_indices = np.append(word_indices, i + 1)

        # ==========================================================
    print('==================')

    return word_indices
