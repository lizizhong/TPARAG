# encoding = utf-8
# the code is based on the source code from https://github.com/facebookresearch/contriever

import sys
import json
import argparse
import logging
from tqdm import tqdm
from functools import reduce
from options import get_options
from typing import List
from tqdm import tqdm

from src.retriever import Contriever, DualEncoderRetriever, UntiedDualEncoderRetriever

import transformers
from transformers import AutoProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import torch
import torch.nn as nn
import numpy as np

from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BERT_MAX_SEQ_LENGTH: int = 512

def set_parser_options(parser: argparse.Namespace, passed_args: List[str]) -> argparse.ArgumentParser:
    all_args = [
        "--query_side_retriever_training",
    ] + passed_args

    return parser.parse_args(all_args)


class Retriever(nn.Module):
    def __init__(self, opt, retriever_model_path, base_retriever_model_path):
        super(Retriever, self).__init__()

        self.opt = opt
        self.retriever_model_path = retriever_model_path
        self.base_retriever_model_path = base_retriever_model_path

        self.retriever, self.tokenizer = self._load_retriever()
        self.retriever.cuda()


    def _load_retriever(self, retriever_is_untied=True):
        contriever_encoder = Contriever.from_pretrained(self.retriever_model_path)
        retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_retriever_model_path)

        if retriever_is_untied:
            retriever = UntiedDualEncoderRetriever(self.opt, contriever_encoder)
        else:
            retriever = DualEncoderRetriever(self.opt, contriever_encoder)

        return retriever, retriever_tokenizer
    
    def _to_cuda(self, tok_dict):
        return {k: v.cuda() for k, v in tok_dict.items()}

    def retriever_tokenize(self, query):
        if self.tokenizer:
            query_enc = self.tokenizer(
                query,
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = self._to_cuda(query_enc)
        else:
            query_enc = None
        return self._to_cuda(query_enc)
    

    def tokenize_passages(self, passages):
        bsz = len(passages)
        n = max([len(example) for example in passages])
        batch = [example + [""] * (n - len(example)) for example in passages]
        batch = reduce(lambda a, b: a + b, passages)

        retriever_tok = self.tokenizer(
                batch,
                padding="max_length",
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                return_tensors="pt",
                truncation=True,
            )
        retriever_tok = self._to_cuda(retriever_tok)
        return retriever_tok

    def tokenize(self, query):
        query_enc = self.retriever_tokenize(query)
        return query_enc

    def retriever_tokenize(self, query):
        if self.tokenizer:
            query_enc = self.tokenizer(
                query,
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = self._to_cuda(query_enc)
        else:
            query_enc = None
        return self._to_cuda(query_enc)

    @torch.no_grad()
    def retriever_embed(self, query, passages):
        bsz = len([query])
        query_enc = self.tokenize([query])
        query_emb = self.retriever(**query_enc, is_passages=False)

        retriever_tokens = self.tokenize_passages([passages])
        retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}
        passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
        passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
        retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])

        return retriever_score
    

class QueryPassage(Dataset):
    def __init__(self, data_file):
        super(QueryPassage, self).__init__()

        self.data_file = data_file
        self.data_instances = self._read_data()

    def _read_data(self):
        instances = {}
        with open(self.data_file, 'r') as j:
            for line in j:
                instances.update(json.loads(line))
        
        total_items = []
        for key, value in tqdm(instances.items()): 
            item_dict = {}
            query, malicious_passages = value['question'], value['malicious_passages']
            # query = value['question']
            passages = value['passages']
            if 'answer' in value:
                answers = value['answer']
            else:
                answers = value['answers']
            
            passage_list = []
            for passage in passages:
                if isinstance(passage, dict):
                    text = (passage['title'] + ' ' + passage['text']).strip()
                else:
                    text = passage.strip()
                passage_list.append(text)

            malicious_passage_list = []
            for passage in malicious_passages:
                content = passage.strip()
                malicious_passage_list.append(content)
            item_dict['index'], item_dict['question'],  item_dict['answers'] = key, query, answers
            item_dict['passages'] = passage_list
            item_dict['malicious_passages'] = malicious_passage_list
            total_items.append(item_dict)

        return total_items

    def __len__(self):
        return len(self.data_instances)
    
    def __getitem__(self, index):
        item = self.data_instances[index]
        item_index = item['index']
        query, malicious_passages = item['question'], item['malicious_passages']
        query = item['question']
        passages = item['passages']
        answer = item['answers']

        return item_index, query, answer, passages, malicious_passages
        # return item_index, query, answer, malicious_passages, passages
        # return item_index, query, answer, passages
    


class RougeScorer(object):
    def __init__(self, metric_type):
        self.metric_type = metric_type
        if metric_type in ['rouge1', 'rouge2', 'rougeL']:
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_score(self, query, passages):
        scores_list = []
        for passage in passages:
            scores = self.scorer.score(query, passage)
            scores_list.append(scores[self.metric_type].fmeasure)

        return scores_list
    

class SentenceBERTScore(object):
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name)

    def calculate_score(self, query, passages):
        if self.model_name == 'hkunlp/instructor-large':
            query_instruction = (
                    "Represent the Wikipedia question for retrieving supporting documents: "
            )
            corpus_instruction = "Represent the Wikipedia document for retrieval: "
            query_embedding = self.model.encode(query, prompt=query_instruction)
            corpus_embeddings = self.model.encode(passages, prompt=corpus_instruction)
            similarities = cos_sim(query_embedding, corpus_embeddings)
        else:
            query_embedding = self.model.encode(sentences=query)
            passage_embedding = self.model.encode(sentences=passages)
            similarities = self.model.similarity(query_embedding, passage_embedding)
        
        return similarities
    

def write_dictfile(retrieved_dict_list, write_filename):
    with open(write_filename, 'w') as writer:
        for item in retrieved_dict_list:
            json.dump(item, writer)
            writer.write('\n')
    writer.close()


def read_json(filename):
    instances = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as j:
        for line in j:
            instances.append(json.loads(line))

    return instances


def generate_rouge_score(opt):
    qp_datas = QueryPassage(data_file=opt.data_filename)
    rouge1_score = RougeScorer(metric_type='rouge1')
    rouge2_score = RougeScorer(metric_type='rouge2')
    rougeL_score = RougeScorer(metric_type='rougeL')
    retrieved_score_dict_list = []
    for i in tqdm(range(qp_datas.__len__())):
        retrieved_score_dict = {}
        item_index, query, malicious_passages, passages = qp_datas[i]
        retrieved_score_dict['index'] = item_index
        retrieved_score_dict['query'] = query
        retrieved_score_dict['passages'] = passages
        retrieved_score_dict['updated_malicious_passages'] = malicious_passages
        rouge1_score_value = rouge1_score.calculate_score(query=query, passages=malicious_passages)
        rouge2_score_value = rouge2_score.calculate_score(query=query, passages=malicious_passages)
        rougeL_score_value = rougeL_score.calculate_score(query=query, passages=malicious_passages)
        retrieved_score_dict['rouge1_score'] = rouge1_score_value
        retrieved_score_dict['rouge2_score'] = rouge2_score_value
        retrieved_score_dict['rougeL_score'] = rougeL_score_value
        retrieved_score_dict_list.append(retrieved_score_dict)

    write_dictfile(retrieved_dict_list=retrieved_score_dict_list,
                    write_filename=opt.retrieved_rouge_file)
    logging.info('rouge metric done')


def generate_sentence_bert_score(opt):
    qp_datas = QueryPassage(data_file=opt.data_filename)
    sentence_bert = SentenceBERTScore()
    retrieved_score_dict_list = []
    for i in tqdm(range(qp_datas.__len__())):
        retrieved_score_dict = {}
        item_index, query, malicious_passages, passages = qp_datas[i]
        retrieved_score_dict['index'] = item_index
        retrieved_score_dict['query'] = query
        retrieved_score_dict['passages'] = passages
        retrieved_score_dict['updated_malicious_passages'] = malicious_passages
        sentence_bert_score = sentence_bert.calculate_score(query=query,
                                                            passages=malicious_passages)
        sentence_bert_score = sentence_bert_score.detach().cpu().numpy().tolist()
        retrieved_score_dict['sentence_bert_score'] = sentence_bert_score[0]
        retrieved_score_dict_list.append(retrieved_score_dict)

    write_dictfile(retrieved_dict_list=retrieved_score_dict_list,
                write_filename=opt.retrieved_bert_file)
    logging.info('rouge metric done')

    
def generate_retriever_score(opt):
    qp_datas = QueryPassage(data_file=opt.data_filename)
    # for index, data in enumerate(qp_datas):
    #     print(index, data)
    # test_dataloader = DataLoader(qp_datas, batch_size=1, shuffle=False)
    retriever = Retriever(opt=opt, retriever_model_path=opt.retriever_model, 
                          base_retriever_model_path=opt.retriever_model)
    # qp_pairs = qp_datas.__getitem__(1)

    retrieved_score_dict_list = []
    for i in tqdm(range(qp_datas.__len__())):
        retrieved_score_dict = {}
        # item_index, query, malicious_passages, passages = qp_datas[i]
        item_index, query, answer, passages, malicious_passages = qp_datas[i]
        retrieved_score_dict['index'] = item_index
        retrieved_score_dict['question'] = query
        retrieved_score_dict['answer'] = answer
        retrieved_score_dict['passages'] = passages
        retrieved_score_dict['malicious_passages'] = malicious_passages
        malicious_retriever_score = retriever.retriever_embed(query=query,
                                                passages=malicious_passages)
        malicious_retriever_score = malicious_retriever_score.detach().cpu().numpy().tolist()
        retrieved_score_dict['malicious_score'] = malicious_retriever_score
        retriever_score = retriever.retriever_embed(query=query,
                                                passages=passages)
        retriever_score = retriever_score.detach().cpu().numpy().tolist()
        retrieved_score_dict['retriever_score'] = retriever_score
        retrieved_score_dict_list.append(retrieved_score_dict)

    write_dictfile(retrieved_dict_list=retrieved_score_dict_list,
                    write_filename=opt.retrieved_file)
    logging.info('done')
        

def print_filted_cases(opt, score_threshold=1.042):
    cnt = 0
    test_instances = read_json(opt.retrieved_file)
    score_list = []
    for item in test_instances:
       index, malicious_score = item['index'], item['score'][0]
    #    index, malicious_score = item['index'], item['malicious_score'][0]
       score_list.extend(malicious_score)
       for score_pos, score in enumerate(malicious_score):
           if score < score_threshold:
               cnt += 1
               logging.info('above threshold, index: {}, position: {}'.format(index, score_pos))
    
    logging.info('total count: {}'.format(cnt))
    score_list = np.array(score_list)
    logging.info('score percentile 20% {}'.format(np.percentile(score_list, 20)))
    logging.info('score percentile 50% {}'.format(np.percentile(score_list, 50)))
    logging.info('score percentile 80% {}'.format(np.percentile(score_list, 80)))
    print('cnt: ', cnt)


def z_score(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        raise ValueError('Standard deviation is zero, z-scores cannot be computed.')
    
    return (arr - mean) / std


def read_score_list(opt, score_type='retrieve'):
    if score_type == 'retrieve':
        test_instances = read_json(opt.retrieved_file)
        score_key_name = 'malicious_score'
    elif score_type == 'rouge':
        test_instances = read_json(opt.retrieved_rouge_file)
        score_key_name = 'rouge2_score'
    else:
        test_instances = read_json(opt.retrieved_bert_file)
        score_key_name = 'sentence_bert_score'

    score_list = np.array([])
    for item in test_instances:
        index, score = item['index'], item[score_key_name]
        if len(score) == 1:
            score = score[0]
        # z_scored_list = z_score(score)
        # score_list.extend(score)
        z_scored_list = np.array(score)
        score_list = np.concatenate((score_list, z_scored_list), axis=0)

    # return z_score(score_list)
    return np.array(score_list)
    


def analysis_similarity_corr(opt):
    retriever_score = read_score_list(opt=opt)
    rouge_score = read_score_list(opt=opt, score_type='rouge')
    bert_score = read_score_list(opt=opt, score_type='sentence_bert')

    plt.figure(figsize=(10, 6))

    x = np.arange(len(retriever_score))
    # Plot the three arrays as lines
    plt.plot(x, retriever_score, label='retriever_score', linewidth=2)
    plt.plot(x, rouge_score, label='rouge_score', linewidth=2)
    plt.plot(x, bert_score, label='bert_score', linewidth=2)

    # Add labels and title
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Line Plot of Three Arrays', fontsize=14)

    # Add a legend
    plt.legend()

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # save the plot
    plt.savefig('score_corr.png')

    rouge_correlation = np.corrcoef(retriever_score, rouge_score)[0, 1]
    bert_correlation = np.corrcoef(retriever_score, bert_score)[0, 1]
    
    print(f"Rouge2 Pearson Correlation Coefficient: {rouge_correlation}")
    print(f"BERT Pearson Correlation Coefficient: {bert_correlation}")
    
    


                    
