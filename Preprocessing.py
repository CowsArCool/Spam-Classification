# Base Packages
import numpy as np
import pandas as pd

# Tokenizattion
import spacy

# Text Processing
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences


class SpamDataset (Dataset):
    def __init__(self, df, max_len=80,
                 spacy_tokenizer=spacy.load('en_core_web_sm')):
        """Initialize the vocabulary object

        Args:
            df (pd.DataFrame): [pandas dataframe with two columns, 1 named levels the other as text]
            max_len (int, optional): [maximum length for a sequence ]. Defaults to 80.
            spacy_tokenizer ([type], optional): [description]. Defaults to spacy.load('en_core_web_sm').
        """

        self.trg = df['labels'].copy()
        self.trg = self.label_encode()

        self.max_len = max_len

        # self.trg[2] = 'spam'
        # print (self.trg[5])

        self.src = df['text'].copy()

        self.spacy_tokenizer = spacy_tokenizer

        self.src = self.tokenize_all(self.src)

        self.stoi = {'<pad>': 0}
        self.itos = {0: '<pad>'}

        self.build_vocab()

        self.src = self.int_encode_all()

        self.src = pad_sequences(self.src,
                                 padding='post',
                                 truncating="post",
                                 maxlen=max_len,
                                 value=0)
        # print (self.src[2])
        # print (self.trg[2])

    def get_output_dim(self):
        return 1

    def tokenize(self, text):
        return [token.text.lower() for token in self.spacy_tokenizer.tokenizer(text)]

    def tokenize_all(self, texts):
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(self.tokenize(text))

        return tokenized_texts

    def build_vocab(self):
        sentences = self.src

        frequencies = {}

        for sentence in sentences:
            for word in sentence:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

        # print (list(frequencies.items())[:10])
        sorted_frequencies = dict(
            sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
        # print (list(sorted_frequencies.items())[:10])
        # print (frequencies.items())

        idx = 1
        for word, freq in sorted_frequencies.items():
            if word not in self.stoi:

                self.stoi[word] = idx
                self.itos[idx] = word

                idx += 1

                # if idx <= 10:
                #     print (word)
                #     print (self.stoi)

    def int_encode_all(self):
        tokenized_sentences = self.src

        for sentence in range(len(tokenized_sentences)):
            tokenized_sentences[sentence] = self.int_encode(
                tokenized_sentences[sentence])

        return tokenized_sentences

    def int_encode(self, sentence):
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in sentence
        ]

    def label_encode(self):
        self.trg
        labels = np.zeros((len(self.trg)), dtype=float)

        for i, label in enumerate(self.trg):
            if label == 'spam':
                # labels[i] = np.array ([1 , 0])
                labels[i] = np.array(1.0)

            elif (label == 'ham'):
                # labels[i] = np.array ([0 , 1])
                labels[i] = np.array(0.0)

            else:
                print('\033[1m" + LABEL DOES NOT MATCH HAM OR SPAm + "\033[0m')

        return labels

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        return {"trg": self.trg[index],
                "src": self.src[index]
                }

    @staticmethod
    def initialize_data(csv_files):
        df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])

        df.drop(df.columns[[np.arange(2, 5)]], axis=1, inplace=True)
        df.rename(columns={'v1': 'labels', 'v2': 'text'}, inplace=True)

        return df


class AmazonReviewsDataset ():
    def __init__(self, df, max_len=80, frequency_threshold=2,
                 spacy_tokenizer=spacy.load('en_core_web_sm')) -> None:

        self.spacy_tokenizer = spacy_tokenizer
        self.frequency_threshold = frequency_threshold

        sentences = df['text'].map(self.tokenize_sentence, na_action='ignore')
        self.stoi, self.itos = self.build_vocab(sentences)
        encoded_sentences = sentences.map(self.int_encode)

        self.src = pad_sequences(encoded_sentences,
                                 padding='post',
                                 truncating="post",
                                 maxlen=max_len,
                                 value=0)

        self.trg = self.build_int_targets(df['label'])

    def int_encode(self, sentence):
        return [self.stoi[word] if word in self.stoi.keys() else 1 for word in sentence]

    def tokenize_sentence(self, sentence):
        return [token.text.lower() for token in self.spacy_tokenizer.tokenizer(sentence)]

    def build_vocab(self, sentences):
        frequencies = {}
        stoi = {'<pad>': 0, '<unk>': 1}
        itos = {v: k for k, v in stoi.items()}

        for sentence in sentences:
            for word in sentence:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

        sorted_frequencies = dict(
            sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

        idx = len(stoi)
        for word, freq in sorted_frequencies.items():
            if word not in stoi and freq >= self.frequency_threshold:

                stoi[word] = idx
                itos[idx] = word

                idx += 1

        return (stoi, itos)

    def build_int_targets(self, labels):
        targets = labels.map(lambda x: np.array(1.0) if x == '__label__1' else np.array(
            0.0) if x == '__label__2' else np.array(np.nan))
        assert not np.nan in targets

        return targets.tolist()

    @staticmethod
    def get_output_dim():
        return 1

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return {
            'src': self.src[idx],
            'trg': self.trg[idx]
        }

    @staticmethod
    def initialize_data(csv_files):
        df = pd.concat([
            pd.read_csv(csv_file, delimiter='\n', names=['all'])
            for csv_file in csv_files
        ], ignore_index=True)

        df[['label', 'text']] = df['all'].str.split(' ', expand=True, n=1)
        df.drop(columns=['all'], inplace=True)
        ####### REMOVE SAMPLING######
        # df = df.copy().sample(5000, random_state=1).reset_index(drop=True)
        df = df.iloc[:5000].copy()

        return df
