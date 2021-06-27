import os
from multiprocessing import Pool
from functools import partial
from collections import Counter
from typing import Union, List
import json
from collections.abc import Iterable

import torch
from torch.utils.data import TensorDataset
from torchtext.data import get_tokenizer
import spacy

tokenizer_collection = {'en_core_web_trf': get_tokenizer('spacy', language='en_core_web_trf'),
                        'basic_english': get_tokenizer("basic_english", language='en')
                       }
stop_words = spacy.load('en_core_web_trf').Defaults.stop_words


class SkipGram(TensorDataset):
    def __init__(self, corpus_files: Union[str, List[str]], language: str='en_core_web_trf', min_freq=12, num_ns=5, use_cache=True, reload=False):
        r"""Build a Skip-gram style dataset
        This class is the child of torch.utils.data.TensorDataset. 

        Parameters
        ----------
        corpus_files : Union[str, List[str]]
            The paths to the corpus files. glob-styled input is acceptable.
        language : str
            Choose a tokenizer. Available: 'en_core_web_trf', 'basic_english'.
        min_freq : int
            Minimum frequency of appearing to keep that word in the dictionary instead of treating it as <UNK>.
        num_ns : int
            The number of negative sampling.
        use_cache : bool
            Whether to cache processed data or use existing cache in the current folder.
        reload : bool
            Whether to force reload and re-process the corpus file. Ignore existing cache.
        
        Returns
        ----------
        torch.utils.data.TensorDataset
        """
        self.language = language
        cached_list = ['cached_dictionary.json', 'cached_target.pt', 'cached_context.pt', 'cached_label.pt', 'cached_id_to_freq.pt', 'cached_id_to_prob.pt']

        if use_cache and not reload and all([os.path.exists(i) for i in cached_list]):
            print("Load from cache")
            self.target = torch.load('cached_target.pt')
            self.context = torch.load('cached_context.pt')
            self.label = torch.load('cached_label.pt')
            self.id_to_freq = torch.load('cached_id_to_freq.pt')
            self.id_to_prob = torch.load('cached_id_to_prob.pt')
            with open('cached_dictionary.json', 'r') as f:
                self.info = json.load(f)
            self.word_to_id = self.info['dictionary']
            self.id_to_word = {int(k): v for k, v in self.info['index'].items()}
        else:
            print("Create dataset from corpus")
            self.cnt = SkipGram.build_counter(corpus_files, language)
            self.word_to_id, self.id_to_word, self.id_to_freq, self.id_to_prob = SkipGram.compile_vocab(self.cnt, min_freq)
            self.info = {}
            self.info['dict_size'] = len(self.id_to_word)
            self.info['dictionary'] = self.word_to_id
            self.info['index'] = self.id_to_word
            self.text_ids = SkipGram.convert_text_to_ids(corpus_files, self.word_to_id, language)
            self.subsampled = False
            pos_samples = self.generate_pos_sample()
            neg_samples = self.generate_neg_sample(pos_samples[:, 1], num_ns)
            all_samples = torch.cat([pos_samples, neg_samples], dim=1)
            self.target = all_samples[:, 0]
            self.context = all_samples[:, 1:]
            self.label = torch.zeros_like(self.context)
            self.label[:, 0] = 1
            if use_cache:
                torch.save(self.target, 'cached_target.pt')
                torch.save(self.context, 'cached_context.pt')
                torch.save(self.label, 'cached_label.pt')
                torch.save(self.id_to_freq, 'cached_id_to_freq.pt')
                torch.save(self.id_to_prob, 'cached_id_to_prob.pt')
                with open('cached_dictionary.json', 'w') as f:
                    json.dump(self.info, f, indent=2)

        self.dict_size = self.info['dict_size']
        super(SkipGram, self).__init__(self.target.unsqueeze(1), self.context, self.label)


    def subsampling(self):
        r"""Call to subsample data"""
        if not self.subsampled:
            rand_var = torch.rand(self.text_ids.shape)
            prob = self.id_to_prob[self.text_ids]
            self.text_ids[rand_var < prob] = 0
            self.subsampled = True


    def generate_pos_sample(self, before=2, after=2, subsampling=True):
        r"""Generate positive examples
        A positive example is a pair of target word and context word.

        Parameters
        ----------
        before : int
            The number of words before target word that are context words.
        after : int
            The number of words after target word that are context words.
        subsampling: bool
            Whether to subsample words.
        
        Returns
        ----------
        torch.Tensor : (N, 2) a row is a pair of words.
        """
        if subsampling:
            self.subsampling()
        text = self.text_ids
        text = text[text != 0]
    
        length = len(text) - before - after
        data = torch.zeros((length, before+after, 2), dtype=torch.int64)
        
        if after == 0:
            data[:,:,0] = text[before:,None]
        else:
            data[:,:,0] = text[before:-after,None]
        for j in range(before):
            data[:,j,1] = text[j:length+j]
        for j in range(after):
            data[:,before+j,1] = text[before+1+j:length+before+1+j]

        return data.reshape(-1, 2)


    def generate_neg_sample(self, context, num_ns):
        r"""Generate negative examples

        Parameters
        ----------
        context : torch.Tensor
            Context word vector (N, ).
        num_ns : int
            The number of negative samples per positive sample.
        
        Returns
        ----------
        torch.Tensor : (N, num_ns)
        """
        context = context.unsqueeze(1)
        number_samples = context.shape[0]
        weights = torch.pow(self.id_to_freq, 0.75)
        noise = torch.multinomial(weights, number_samples * num_ns, True).view(number_samples, num_ns)
        for _ in range(2):
            reptitive_select = torch.sum(context == noise, dim=1) >= 1
            noise_new = torch.multinomial(weights, number_samples * num_ns, True).view(number_samples, num_ns)
            noise[reptitive_select] = noise_new[reptitive_select]
        return noise


    def __call__(self, tokens):
        r"""Tokenize and convert tokens to ids"""
        if isinstance(tokens, str):
            tokens = tokens.lower()
            tokenizer = tokenizer_collection[self.language]
            ids = [self.word_to_id.get(tok, 0) for tok in tokenizer(tokens.strip())]
            return ids
        elif isinstance(tokens, Iterable):
            ids = []
            for i in tokens:
                ids.extend(self(str(i).lower()))
            return ids
        else:
            raise TypeError(f'{type(tokens)} is not supported')
    

    def lookup_words(self, ids: Union[int, List[int]]):
        r"""Find words based on ids"""
        if isinstance(ids, int):
            return [self.id_to_word.get(ids, "<unk>")]
        elif isinstance(ids, Iterable):
            words = []
            for i in ids:
                words.extend(self.lookup_words(int(i)))
            return words
        else:
            raise TypeError(f'{type(ids)} is not supported')


    @staticmethod
    def subsample_probability(frequency, sampling_factor):
        r"""
        Generates a word rank-based probabilistic sampling table
        More information: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/make_sampling_table
        """
        frequency = frequency / torch.sum(frequency)
        prob = (torch.min(torch.FloatTensor([1.0]), torch.sqrt(frequency / sampling_factor) / (frequency / sampling_factor)))
        prob = 1 - prob # 1 - prob is rejection probability
        return torch.clamp(prob, 0, 1)


    @staticmethod
    def _build_counter_process(path, language='en_core_web_trf'):
        r"""Count word's frequency"""
        tokenizer = tokenizer_collection[language]
        cnt = Counter()
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                tokens = tokenizer(line.strip())
                cnt.update(tokens)
        return cnt


    @staticmethod
    def build_counter(files, language='en_core_web_trf'):
        r"""Count word's frequency from multiple files"""
        if isinstance(files, str):
            return SkipGram._build_counter_process(files, language)

        n = len(files)
        num_worker = min(12, n, os.cpu_count())
        cnt = Counter()
        process_func = partial(SkipGram._build_counter_process, language=language)
        with Pool(num_worker) as p:
            for curr in p.imap(process_func, files):
                cnt += curr
        return cnt


    @staticmethod
    def compile_vocab(counter, min_freq=12, threshold=1e-5):
        r"""Compile the counter (collections.Counter) to get vocabulary and related informantion

        Parameters
        ----------
        counter : collections.Counter
            Mapping word to frequency.
        min_freq : int
            Minimum frequency of appearing to keep that word in the dictionary instead of treating it as <UNK>.
        threshold : float
            Subsampling factor. check: subsample_probability(frequency, sampling_factor).
        
        Returns
        ----------
        dict : Mapping word to id.

        dict : Mapping id to word.

        torch.Tensor : Mapping id to frequency.

        torch.Tensor : Mapping id to subsampling table.
        """
        word_to_id = {"<unk>":0}
        id_to_word = {0:"<unk>"}
        id_to_freq = [1]
        id = 1
        for word, freq in counter.items():
            if word == "<unk>":
                continue
            elif freq > min_freq and word not in stop_words:
                word_to_id[word]=id
                id_to_word[id] = word
                id_to_freq.append(freq)
                id += 1
            else:
                word_to_id[word] = 0
        id_to_freq = torch.tensor(id_to_freq, dtype=torch.float32)
        id_to_prob = SkipGram.subsample_probability(id_to_freq, threshold)
        return word_to_id, id_to_word, id_to_freq, id_to_prob


    @staticmethod
    def _text_to_id(path, word_to_id, language: str='en_core_web_trf'):
        r"""Convert text into ids"""
        tokenizer = tokenizer_collection[language]
        ids = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                tokens = tokenizer(line.strip())
                for tok in tokens:
                    ids.append(word_to_id.get(tok, 0))
        return ids
    

    @staticmethod
    def convert_text_to_ids(files, word_to_id, language: str='en_core_web_trf'):
        r"""Convert text into ids from multiple files"""
        if isinstance(files, str):
            ids = SkipGram._text_to_id(files, word_to_id, language)
            return torch.LongTensor(ids)

        n = len(files)
        num_worker = min(12, n, os.cpu_count())
        all_ids = []
        process_func = partial(SkipGram._text_to_id, word_to_id=word_to_id, language=language)
        with Pool(num_worker) as p:
            for ids in p.imap(process_func, files):
                all_ids.extend(ids)
        return torch.LongTensor(all_ids)
