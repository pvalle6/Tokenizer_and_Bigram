"""
    This a Byte Pair Encoding Algorithm and a Bigram Prediction Algorithm.

    Written by Peter Vallet January 2024
    https://www.linkedin.com/in/peter-v-334609211/
"""

# Only default packages are imported per instructions
import argparse
import pickle as pkl
import re
import random

parser = argparse.ArgumentParser()

parser.add_argument("selector", help="Selector",
                    choices=["train_bigram", "predict_bigram", "train_bpe", "tokenize"])

parser.add_argument("--data", help="Filepath of Data Corpus to train models on,")
parser.add_argument("--save", help="Filepath to pickle bigram or BPE model.")
parser.add_argument("--load", help="Filepath to trained bigram or BPE model")
parser.add_argument("--word", help="Specifies first word used in predict_bigram")
parser.add_argument("--nwords", help="specifies number of words to predict using predict_bigram")
parser.add_argument("--text", help="specifies a string to be tokenized")
args = parser.parse_args()


class BigramModel:
    """
    This the class implementing a Bigram Prediction Model
    """

    def __init__(self):
        """
        This is the constructor of the BigramModel class.
        """

        self.word_follow_dict = dict()

    def train(self, text: str):
        """
        This function accepts a string of data on which the bigram model
        can be trained. The function splits provided strings into separate
        words (including periods as a separate word), identifies the unique
        words in the corpus, identifies which other words can follow each unique
        word, and their frequencies.

        :param text: The corpus of text that the model will train on
        """

        def get_word_follow_dict(input_text: str):
            """
            Generates a dictionary of unique words in corpus as keys and dictionaries of words and their freq that
            follow them as the values.


            :param input_text:
            :return unique_words: a dictionary of unique words in corpus as keys and another dictionary of
            unique words that appear as forward adjacent words in the corpus with their frequency value.

            e.g. {"white": {"whale: n, ...}}
            """

            # assumes that all words in the corpus can be either found by space or by splitting a period
            list_words = input_text.split(" ")

            """
            This function enables the splitting of periods as separate words in the corpus as instructed.
            """
            for i in range(len(list_words) - 1):
                if list_words[i].find(".") != -1:
                    if list_words[i] != ".":
                        w = list_words[i]
                        v_loc = list_words[i].find(".")
                        split = [w[:v_loc], '.']

                        list_words[i] = split[0]
                        list_words.insert(i + 1, split[1])

            r"""
            It came to my attention that this does not remove the \n character in the corpus; it causes a combined word 
            effect to happen where '[word 1]\all[word2]' occurs

            The loop below fixes this error
            """

            for i in range(len(list_words) - 1):
                if list_words[i].find("\n") != -1:
                    if list_words[i] != "\n":
                        w = list_words[i]
                        v_loc = list_words[i].find("\n")
                        split = [w[:v_loc], '.']

                        list_words[i] = split[0]

            # creates a dictionary to hold the unique words and their following word's freq
            unique_words = dict()

            # This loop populates the dictionary with the unique words and their following words
            for i in range(len(list_words) - 1):
                if list_words[i] not in unique_words:
                    unique_words.update({list_words[i]: {list_words[i + 1]: 1}})
                else:
                    if list_words[i + 1] in unique_words[list_words[i]]:
                        unique_words[list_words[i]].update(
                            {list_words[i + 1]: unique_words[list_words[i]][list_words[i + 1]] + 1})
                    else:
                        unique_words[list_words[i]].update({list_words[i + 1]: 1})

            return unique_words

        self.word_follow_dict = get_word_follow_dict(text)

    @staticmethod
    def get_probability(main_word: str, next_word: str, wf_dict: dict) -> float:
        """
        Gets the probability of a next_word following main_word; so long as they are in the word_follow dictionary.

        :param main_word: the prior word that other words follow (must be in the word_follow dictionary)
        :param next_word: the posterior word that you wish to examine (must be in the word_follow main_word sub
        dictionary)
        :param wf_dict: the dictionary created in get_word_follow_dict() that contains all the prior words
        and their posterior values
        :return: the probability of "bag of words" model assuming that the posterior word is chosen at random
        based on the frequency it appears
        """

        next_word_count = wf_dict[main_word][next_word]  # returns the frequency of the selected word

        total_followed = 0  # variable that counts all prior words and their frequency
        for f_word in wf_dict[main_word]:
            total_followed = total_followed + wf_dict[main_word][f_word]

        return next_word_count / total_followed  # returns a basic probability of (specific word / population of words)

    def predict_next_word(self, start_word: str, numb_pred: int) -> list:
        """
        This function accepts the current word as an input and samples
        the next word from the probability distribution learned through
        training. The function additionally checks for the existence of
        provided word in the corpus.

        :param start_word: the word you want to start your predictions with
        :param numb_pred: the number of sequential predictions to be made
        :return list_ret_words: the list of the sequence of predictions
        """

        list_ret_words = list()
        word_dict = self.word_follow_dict

        try:

            # iterates over the number of prediction sequences requested
            for _ in range(numb_pred):
                """
                This logic constructs a probability distribution through the creation of intervals based on 
                get_probability. It then samples based on a randomly generated number which word to present to the list
                and begin a new prediction.
                """
                percent_one = 1
                prob_dict = dict()
                for f_word in word_dict[start_word]:
                    next_prob = self.get_probability(start_word, f_word, word_dict)
                    prob_dict.update(
                        {f_word: [percent_one, percent_one - next_prob]})
                    percent_one = percent_one - next_prob

                chance = random.random()
                retword = ''
                for next_word in prob_dict:  # checks for one word that fits the prob distribution
                    if (prob_dict[next_word][0]) > chance >= (prob_dict[next_word][1]):
                        retword = next_word

                start_word = retword
                list_ret_words.append(retword)
        except KeyError:
            print("Word not found in Corpus")

        return list_ret_words


class BPEAlgorithm:
    """
    This is the class the tokenizing Byte Pair Encoder.
    """

    def __init__(self):
        """
        This is the constructor of the BPE algorithm class.
        """
        self.vocabulary = list()
        self.token_ids = dict()

    def train(self, corpus_text: str, k: int = 500):
        """
        This function accepts preprocessed text data as input and
        implements the BPE learner algorithm.

        :param corpus_text: This is the text that the model will be trained on.
        :param k: This is the maximum number of recombination to consider
        """

        def word_rank(text_data, stop_condition: int):
            """
            This function takes in a corpus, and then creates a vocabulary based on the BPE Algorthm.

            The vocabulary is built in a pyramid like fashion whereby the original tokens are single digit, which
            are identified by their initial appearance in a text. Then larger tokens are identified as possible byte
            pairs to be constructed from smaller tokens already identified. In this use, every high order token such as
            four character token contains a mixture of at least the previous order (third order) and possibly other
            lesser orders.

            :param text_data: This is the text that the vocabulary will be extracted from
            :param stop_condition: This is the maximum number of recombination to consider.
            :return wip_vocab: This is the vocabulary for the model.
            """

            wip_vocab = []  # list that holds the vocabulary for use in the function

            processed_split = BPEAlgorithm.add_eow_character(text_data)

            print("TRAINING MODEL (this may take a long time)")
            """
            This code finds every unique single character and adds it to the vocabulary
            """
            wip_vocab.append('|')
            wip_vocab.append('[UNK]')
            for char in processed_split:
                if char not in wip_vocab:
                    wip_vocab.append(char)

            # This splits the entire corpus into individual characters
            list_char = list(BPEAlgorithm.add_eow_character(text_data))
            recombine = len(wip_vocab)
            max_freq = 5  # max_freq dictates that nothing will add to the vocabulary if it does not occur 5 times
            """"
            the stop conditions are: k recombination and the highest frequency of the most common pair must be
            greater than 5
            """

            """
            This takes forever to train
            """
            while recombine < stop_condition and max_freq >= 2:
                pair_freq = dict()  # pair_freq is a dictionary that holds tuples of pairs of strings and their freq

                for i in range(len(list_char) - 1):  # grabs every pair from the corpus
                    if (i + 1) < len(list_char):  # checks if the pair starts with |
                        if list_char[i] != '|':

                            if (list_char[i], list_char[i + 1]) in pair_freq:  # checks if the tuple exists in pair_freq
                                pair_freq.update(
                                    {(list_char[i], list_char[i + 1]): pair_freq[(list_char[i], list_char[i + 1])] + 1})
                            else:
                                pair_freq.update({(list_char[i], list_char[i + 1]): 1})

                """
                sorts the dictionary by going to a list value
                found how to do this functional stuff sort dictionary by stackoverflow
                """
                pair_freq_list = sorted(pair_freq.items(), key=lambda item: item[1])

                max_freq = pair_freq_list[-1][1]  # finds the highest freq combination to enforce stop conditions
                # grabs the highest pair tuple and makes it into a string
                high_pair_str = pair_freq_list[-1][0][0] + pair_freq_list[-1][0][1]

                pair_freq_list.pop()  # removes this highest pair tuple from the sorted list

                wip_vocab.append(high_pair_str)  # adds highest pair string to the vocabulary

                # replaces in the list of characters, this highest pair string where it does exist
                for i in range(len(list_char) - 1):
                    if (i + 1) < len(list_char):
                        if (list_char[i] + list_char[i + 1]) == high_pair_str:
                            list_char.pop(i)
                            list_char[i] = high_pair_str

                # iterates to enforce stopping conditions
                recombine = recombine + 1
                print(str(recombine) + "/500")
            return wip_vocab

        self.vocabulary = word_rank(text_data=corpus_text, stop_condition=k)
        new_token_id = 1

        for vocab in self.vocabulary:  # this generates a unique token for our vocabulary
            self.token_ids.update({vocab: new_token_id})
            new_token_id = new_token_id + 1

    @staticmethod
    def add_eow_character(text: str) -> str:
        """
        Add eow character to each regex identified word in the corpus
        :param text: This is the text to process.
        :return text_with_eow: Processed Text.
        """
        # regex code that I took from the lecture 2 slides
        tokens = re.findall(r'\b\w+\b|[!?;.,]', text)

        text_with_eow = ''.join([token + '|' for token in tokens])

        return text_with_eow

    def tokenize(self, string: str):
        """
        This function accepts a string and uses the trained vocabulary
        to tokenize the provided string. The function returns a token and
        a token ID corresponding to the token.

        :param string: This is the provided string to use trained vocabulary to tokenize
        :return tokenized_split, str_token_ids: This is the tokenized list and list of token ids corresponding to the
        provided strings.
        """
        # processed string with EOW characters
        p_string = self.add_eow_character(string)

        tokenized_split = [p_string]  # This dict is the basis of the functionality of the script below

        for character in tokenized_split[-1]:  # this searches through string for any unknown characters and replaces it
            if character not in self.vocabulary:
                tokenized_split[-1][tokenized_split[-1].find(character)] = self.vocabulary[1]

        """
        This algorithm searches through the provided processed string, tokenizes it using tokens in order they were
        learned, which naturally is single character tokens to multiple character tokens, by frequency
        """

        for letter in tokenized_split[-1]:
            for vocab in self.vocabulary:
                if letter.find(vocab) != -1:
                    w = tokenized_split[-1]
                    tokenized_split.pop()
                    v_loc = w.find(vocab)
                    len_v = len(vocab)

                    if v_loc == 0:
                        split = [vocab, w[len_v:]]
                    else:
                        split = [w[:v_loc], vocab, w[len_v + v_loc:]]

                    tokenized_split = tokenized_split + split

        """
        This loop checks if tokens are able build larger tokens as the vocabulary is naturally built in a pyramid shape.
        That is that larger vocabulary tokens are always built from smaller tokens orders
        """
        for vocab in self.vocabulary:  # This is the part that checks to see if a larger token can be substituted
            for i in range(len(tokenized_split) - 1):
                if i + 1 < (len(tokenized_split)):
                    if (tokenized_split[i] + tokenized_split[i + 1]) == vocab:
                        tokenized_split[i] = vocab
                        tokenized_split.pop(i + 1)

        str_token_ids = []  # constructs a list of token ids are found from the vocabulary
        for token in tokenized_split:
            str_token_ids.append(self.token_ids[token])

        return tokenized_split, str_token_ids


"""
    PROGRAM FUNCTIONALITY
"""
print("----------------------------------------------------------------------------------------------------")
print("Program Started \n Peter Vallet 2024 \n HNRS 3035: Design and Business Applications of Large Language Models.")

corpus = None  # corpus is a placeholder variable for holding the entire corpus of text later on if needed
p_corp = None  # p_corp is a placeholder variable for the processed corpus of text for use in the BPE
bigram_model = None  # models are placeholder variables for pickling of either model depending on args
bpe_model = None

if args.load is not None and args.selector == "predict_bigram":
    with open(args.load, "rb") as f:
        bigram_model = pkl.load(f)
    print("Model loaded")

if args.load is not None and args.selector == "tokenize":
    with open(args.load, "rb") as f:
        bpe_model = pkl.load(f)

    print("Model loaded")


if args.data is not None:
    with open(args.data, "r", encoding="utf8") as f:
        corpus = f.read()
    print("Corpus read in to variable")

if args.selector == "train_bigram" and corpus is not None:  # trains a BigramModel on a corpus of text
    bigram_model = BigramModel()
    bigram_model.train(corpus)


if args.selector == "train_bpe" and corpus is not None:  # trains a BPEAlgorithm on a processed corpus of text

    bpe_model = BPEAlgorithm()
    bpe_model.train(corpus)

if args.selector == "tokenize" and bpe_model is not None:  # uses loaded a BPEAlgorithm to tokenize string
    print("Tokenizing....")

    tk_string, tk_ids = bpe_model.tokenize(args.text)
    print(tk_string)
    print(tk_ids)


if args.selector == "predict_bigram" and bigram_model is not None:  # checks if mode is predict_bigram and predicts word
    print("NOW PREDICTING WORDS (this takes less than a minute)")
    print(bigram_model.predict_next_word(args.word, int(args.nwords)))


if args.save is not None:  # checks if the user provided a file path to save model as a pickle
    with open(args.save, "wb") as save_file:

        if args.selector == "train_bigram":
            pkl.dump(bigram_model, save_file)
        if args.selector == "train_bpe":
            pkl.dump(bpe_model, save_file)

    print("Model saved")

print("Program Ended")
print("----------------------------------------------------------------------------------------------------")

""" 
    END PROGRAM 
"""
