#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import re
from collections import defaultdict, namedtuple
from enum import Enum

import ftfy
import spacy
from tqdm import trange


class GPTTokenizer:
    word_end = '</w>'

    @staticmethod
    def _text_standardize(text):
        """
        fixes some issues the spacy tokenizer had on books corpus
        also does some whitespace standardization
        """

        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub('\s*\n\s*', ' \n ', text)
        text = re.sub('[^\S\n]+', ' ', text)
        text = text.strip()

        return text

    def __init__(self):
        self.spacy_tokenizer = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

    def tokenize(self, text):
        text = ftfy.fix_text(text)
        text = GPTTokenizer._text_standardize(text)
        tokens = [t.text.lower() for t in self.spacy_tokenizer(text)]
        tokens = [tuple(token[:-1]) + (token[-1] + GPTTokenizer.word_end,) for token in tokens]

        return tokens

    def detokenize(self, bpe_tokens):
        return ''.join(bpe_tokens).replace(GPTTokenizer.word_end, ' ')


class GPT2Tokenizer:
    @staticmethod
    def _bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """

        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]

        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
                
        cs = [chr(n) for n in cs]
        mapping = dict(zip(bs, cs))

        return mapping

    def __init__(self):
        self.byte_encoder = GPT2Tokenizer._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def tokenize(self, text):
        tokens = []
        tokens = [''.join([self.byte_encoder[b] for b in t.encode('utf-8')]) for t in re.findall(self.pat, text)]
        tokens = [tuple(token) for token in tokens]

        return tokens

    def detokenize(self, bpe_tokens):
        text = ''.join(bpe_tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')

        return text


SpecialTokensTuple = namedtuple('SpecTokensTuple', ['pad',
                                                    'bos',
                                                    'eos',
                                                    'info_bos',
                                                    'info_eos',
                                                    'talker1_bos',
                                                    'talker1_eos',
                                                    'talker2_bos',
                                                    'talker2_eos',
                                                    'sent_dialog',
                                                    'info_dialog',
                                                    'talker1_dialog',
                                                    'talker2_dialog'])


class BPEVocab:
    @staticmethod
    def from_files(vocab_path, codes_path, tokenizer, special_tokens):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version:'):
                codes = codes[1:]
            if len(codes[-1]) == 0:
                codes = codes[:-1]

            codes = [tuple(c.split()) for c in codes if c]

        return BPEVocab(vocab, codes, tokenizer, special_tokens)

    @staticmethod
    def _get_pairs(sequence):
        if len(sequence) < 2:
            return set()

        return set(zip(sequence[:-1], sequence[1:]))

    def __init__(self, vocab, codes, tokenizer, special_tokens):
        assert isinstance(special_tokens, SpecialTokensTuple)

        filtered_special_tokens = [t for t in special_tokens if t not in vocab]
        special_token2id = {t: i for i, t in enumerate(filtered_special_tokens)}
        token2id = {t: i + len(filtered_special_tokens) for t, i in vocab.items()}
        token2id.update(special_token2id)

        self.special_tokens = special_tokens
        self.n_new_tokens = len(filtered_special_tokens)
        self.token2id = token2id
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.tokenizer = tokenizer
        self.cache = {}

        for token_name, token in special_tokens._asdict().items():
            setattr(self, token_name + '_id', self.token2id[token])

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return self.n_new_tokens

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.special_tokens]

    def get_prefix2words(self, words_dict, smoothing_freq=5):
        # map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts)
        prefix2words = defaultdict(dict)
        for i in trange(len(words_dict)):
            word = words_dict[i]
            freq = words_dict.freq[word] + smoothing_freq
            prefix = self._bpe(word)[0]
            prefix2words[prefix].update(dict([(word, freq)]))

        # translate in map of frequency ratios
        for prefix, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = token
        pairs = BPEVocab._get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = BPEVocab._get_pairs(word)

        self.cache[token] = word

        return word

    def string2ids(self, string):
        tokens = self.tokenizer.tokenize(string)
        bpe_tokens = sum([self._bpe(t) for t in tokens], tuple())
        ids = [self.token2id[t] for t in bpe_tokens if t in self.token2id]

        return ids

    def ids2string(self, ids):
        bpe_tokens = [self.id2token[id] for id in ids]
        string = self.tokenizer.detokenize(bpe_tokens)

        return string


SPECIAL_TOKENS = SpecialTokensTuple(pad='<pad>',
                                    bos='<s>',
                                    eos='</s>',
                                    info_bos='<i>',
                                    info_eos='</i>',
                                    talker1_bos='<t1>',
                                    talker1_eos='</t1>',
                                    talker2_bos='<t2>',
                                    talker2_eos='</t2>',
                                    sent_dialog='<sent>',
                                    info_dialog='<info>',
                                    talker1_dialog='<talker1>',
                                    talker2_dialog='<talker2>')


ZERO_SHOT_SPECIAL_TOKENS = SpecialTokensTuple(pad='<pad>',
                                              bos='^',
                                              eos='_',
                                              info_bos='<',
                                              info_eos='>',
                                              talker1_bos='{',
                                              talker1_eos='}',
                                              talker2_bos='[',
                                              talker2_eos=']',
                                              sent_dialog='|',
                                              info_dialog='@',
                                              talker1_dialog='*',
                                              talker2_dialog='#')


class VocabType(Enum):
    GPT = 'gpt'
    GPT2 = 'gpt2'


def get_vocab(vocab_path, codes_path, tokenizer_type=VocabType.GPT, zero_shot_special_tokens=False):
    if tokenizer_type == VocabType.GPT:
        tokenizer = GPTTokenizer()
    elif tokenizer_type == VocabType.GPT2:
        tokenizer = GPT2Tokenizer()
    else:
        assert False

    if zero_shot_special_tokens:
        special_tokens = ZERO_SHOT_SPECIAL_TOKENS
    else:
        special_tokens = SPECIAL_TOKENS

    return BPEVocab.from_files(vocab_path, codes_path, tokenizer, special_tokens)
