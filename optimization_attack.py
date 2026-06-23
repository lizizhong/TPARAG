# encoding = utf-8
import os
import sys
import random
import openai
import argparse
from openai import OpenAI
from tqdm import tqdm
from glob import glob
from options import get_options
from collections import Counter

import numpy as np

from flair.data import Sentence
from flair.nn import Classifier

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from bm25 import BM25_Model

from contriever_filter import Retriever, set_parser_options, SentenceBERTScore, RougeScorer
from generate_passage_utils import read_instances, write_malicious_passage, extract_span_numbers
from generation_attack import generate_passage


OPENAI_API_KEY = None
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# load entity recognition pipeline for attack localization
tagger_ner = Classifier.load('ner')
tagger_pos = Classifier.load('pos')

# load generate language model attacker
model_id = 'Qwen/Qwen2.5-14B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
generation_model = model

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
generation_tokenizer = tokenizer

# args for loading the retriever
options = get_options()
opt = set_parser_options(options.parser, sys.argv[1:])
# white-box setting
retriever = Retriever(opt=opt, retriever_model_path=opt.retriever_model, 
                        base_retriever_model_path=opt.retriever_model)

# black-box setting
sentence_bert = SentenceBERTScore()

# init rouge score for similarity filtering
rouge2_score = RougeScorer(metric_type='rouge2')


def ance_retriever_embed(query, passages):
    '''Use ANCE retriever (Xiong et al., 2020) to calculate the textual similarity scores between query and malicious passage candidates
    Args:
        query: string
        passages: List[string]
    '''
    sentence_bert_score = sentence_bert.calculate_score(query=query,
                                                        passages=passages)
    sentence_bert_score = sentence_bert_score.detach().cpu().numpy().tolist()
    
    return sentence_bert_score


def get_bm25_score(query, malicious_passages):
    '''Use BM25 to calculate the textual similarity scores between query and malicious passage candidates
    Args:
        query: string
        passages: List[string]
    '''

    bm25_model = BM25_Model(documents_list=malicious_passages)
    scores_list = bm25_model.get_documents_score(query=query)
    
    return [scores_list]


def load_classifier(model_checkpoint):
    '''load the pre-trained entity classifier for entity recognition
    Args:
        model_checkpoint: checkpoint file path
    '''
    classifier = pipeline("sentiment-analysis", model=model_checkpoint, device='cuda:0')
    return classifier


def combine_single_substution_process_ner(backup_token, word_text, ner_begin, backup_token_list):
    '''replace the one chosen ner token with one of the candidate tokens
    Args:
        backup_token: dict, replace token candidate dict (e.g., generated token: list of top-k candidate tokens)
        word_text: string, the content of token will be replaced
        ner_begin: the begin index for the token will be replaced
        backup_token_list: list[string], list of backup_token keys

    '''
    updated_token = ''
    in_span = False
    for index, item in enumerate(backup_token_list):
        if index < ner_begin:
            continue

        elif item == word_text:
            key_token = item
            chosen_candidate_token = random.choice(backup_token[index][key_token])
            updated_token += chosen_candidate_token.strip()
            ner_begin = index + 1
            return updated_token, ner_begin
        
        elif item in word_text and not in_span:
            key_token = item
            if len(backup_token[index][key_token]) >= 1:
                chosen_candidate_token = random.choice(backup_token[index][key_token])
            else:
                print(key_token, backup_token[index][key_token])
                chosen_candidate_token = key_token
            updated_token += chosen_candidate_token.strip()
            in_span = True

        elif item in word_text and in_span:
            key_token = item
            chosen_candidate_token = random.choice(backup_token[index][key_token])
            updated_token += chosen_candidate_token.strip()
            if word_text[-len(item):] == item:
                in_span = False
                ner_begin = index + 1
                return updated_token, ner_begin
            
        elif item not in word_text and in_span:
            updated_token = ''
            in_span = False
        else: # item not in word_text_list[0] and not in_span
            continue

    return updated_token, ner_begin


def combine_substution_process_ner(backup_token, word_text, ner_begin, backup_token_list):
    '''ner token substution function
    Args:
        backup token: dict, substution token candidates dict
        word_text: string, the token content that will be replaced
        ner_begin: int, the index of ner token beginning position
        backup_token_list: List[string], the list of token candidates' keys, easy for searching
    Return: 
        update token in the target position: string
        ner_last_new_begin: the position where the last new token replacement conducted
    '''
    word_text_list = word_text.split(' ')
    if len(word_text_list) == 1:
        # if there is only one token in the target entity, just replace it
        updated_word_str, ner_last_new_begin = combine_single_substution_process_ner(backup_token=backup_token,
                                                                        word_text=word_text_list[0],
                                                                        ner_begin=ner_begin,
                                                                        backup_token_list=backup_token_list)

    else:
        # else do iteration
        updated_word_list, ner_last_new_begin = [], ner_begin
        for word_text in word_text_list:
            updated_word, ner_new_begin = combine_single_substution_process_ner(backup_token=backup_token,
                                                                        word_text=word_text,
                                                                        ner_begin=ner_begin,
                                                                        backup_token_list=backup_token_list)
            updated_word_list.append(updated_word)
            ner_last_new_begin = ner_new_begin
        updated_word_str = ' '.join(updated_word_list)

    return updated_word_str, ner_last_new_begin



def combine_substution_process_pos(backup_token, word_text, backup_index, backup_token_list):
    '''pos token substution function
    Args:
        backup token: substution token candidates dict (e.g., generated token: list of top-k candidate tokens)
        word_text: string, the token content that will be replaced
        backup_index: int, the index of candidate tokens list
        backup_token_list: List[string], candidate tokens list
    Return: 
        update token in the target position: string
    '''
    updated_token = ''
    while backup_index < len(backup_token_list) and backup_token_list[backup_index] in word_text:
        key_token = backup_token_list[backup_index]
        backup_candidate_list = backup_token[backup_index][key_token]
        chosen_candidate_token = random.choice(backup_candidate_list)
        updated_token += chosen_candidate_token.strip()
        backup_index += 1

    return updated_token


def target_substution_attack_single_sentence(passage, backup_token, pred_label, pos_substution_rate):
    '''optimization attack for one single passage
    Args:
        passage: string, the input parent malicious passage
        backup_token: dict, the top-k tokens recorded during generation attack (e.g., generated token: list of top-k candidate tokens)
        pred_label: string, the attack position type
        pos_substution_rate: the substution rate for pos type token
    Return:
        updated malicious passages: string
        substution rate: float
    '''
    updated_ner_worddict = {}
    sentence = Sentence(passage)
    tagger_pos.predict(sentence)
    tagger_ner.predict(sentence)
    if 'ner' in sentence.annotation_layers:
        ner_annotation = sentence.annotation_layers['ner']
    else:
        ner_annotation = []
    if 'pos' in sentence.annotation_layers:
        pos_annotation = sentence.annotation_layers['pos']
    else:
        pos_annotation = []
        print(sentence.annotation_layers)
    
    if pred_label != 'POS' and len(ner_annotation) == 0:
        print('invalid vase: pred_label is NER but no accessible NER annotation')
        pred_label = 'POS'
    
    backup_dict_list = []
    for instance in backup_token:
        backup_dict_list.extend(list(instance.keys()))

    substution_cnt = 0
    if pred_label != 'POS':
        # ner substution
        ner_begin = 0
        for word in ner_annotation:
            word_tag = word.data_point.tag
            word_text = word.data_point.text
            span_index = extract_span_numbers(word.unlabeled_identifier)
            if word_tag == pred_label:
                # if this position's token's ner type is the same as the answer pred type, replace it 
                candidate_token, ner_begin = combine_substution_process_ner(backup_token=backup_token, 
                                                word_text=word_text,
                                                ner_begin=ner_begin,
                                                backup_token_list=backup_dict_list)
                updated_ner_worddict[span_index] = candidate_token
                substution_cnt += 1
            else:
                updated_ner_worddict[span_index] = word_text
    
    if updated_ner_worddict:
        ner_span_list = list(updated_ner_worddict.keys())
    else:
        ner_span_list = None

    updated_word_list = []
    pos_index, ner_index, in_span_flag = 0, 0, False
    backup_dict_index = 0
    rand_pos = np.random.random(len(pos_annotation))
    for rand, word in zip(rand_pos, pos_annotation):
        # iterate each token position for token substution
        word_text = word.data_point.text # token content
        if ner_span_list: # see if in ner replacement span
            if ner_index < len(ner_span_list) and pos_index >= ner_span_list[ner_index][0] and \
                pos_index < ner_span_list[ner_index][1] and not in_span_flag:
                in_span_flag = True
                updated_word_list.append(updated_ner_worddict[ner_span_list[ner_index]])
                pos_index += 1
                continue
            elif ner_index < len(ner_span_list) and pos_index >= ner_span_list[ner_index][0] and \
                pos_index < ner_span_list[ner_index][1] and in_span_flag:
                pos_index += 1
                continue
            elif ner_index < len(ner_span_list) and pos_index == ner_span_list[ner_index][1]:
                ner_index += 1
                if ner_index < len(ner_span_list) and pos_index >= ner_span_list[ner_index][0] and \
                pos_index < ner_span_list[ner_index][1]:
                    updated_word_list.append(updated_ner_worddict[ner_span_list[ner_index]])
                    pos_index += 1
                    assert in_span_flag == True
                    continue
                else:
                    in_span_flag = False

        # pos token substution
        if pred_label == 'POS' and rand < pos_substution_rate:
            candidate_token = combine_substution_process_pos(backup_token=backup_token, 
                                                            word_text=word_text, 
                                                            backup_index=backup_dict_index,
                                                            backup_token_list=backup_dict_list)
            updated_word_list.append(candidate_token)
            substution_cnt += 1
        else:
            updated_word_list.append(word_text)
        while backup_dict_index < len(backup_dict_list) and backup_dict_list[backup_dict_index] in word_text:
            backup_dict_index += 1
        pos_index += 1

    updated_passage = ' '.join(updated_word_list)
    updated_passage = updated_passage.replace(' ,', ',')
    updated_passage = updated_passage.replace(' .', '.')

    if len(updated_word_list) == 0:
        return updated_passage, 0
    else:
        return updated_passage, substution_cnt/len(updated_word_list) # sub rate


def get_model_output_logprobs(prompt, answer):

    input_ids = tokenizer(prompt, padding=False, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    answer_ids = tokenizer(answer, padding=False, return_tensors="pt").input_ids
    if 'Qwen' in model_id:
        answer_length = answer_ids.shape[1] # remove begin_of_text tokens
    else:
        answer_length = answer_ids.shape[1]-1 # remove begin_of_text tokens
    if answer_length == 0:
        print(answer)
        answer_length = 1
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    batch = batch[0]
    sum_answer_prob = 0
    answer_prob_list = batch[-answer_length:]
    for token, prob in answer_prob_list:
        sum_answer_prob += prob
    avg_answer_prob = sum_answer_prob/answer_length

    return avg_answer_prob



def get_pred_logprobs(ori_passage, updated_passage_list, question, answer):
    '''calcuate output logits based on llm attacker
    Args:
        ori_passage: string, original retrieved passage
        updated_passage_list: List[string], the malicious passage candidates
        question: string, given query
        answer: string, ground-truth answer
    Return:
        passages_list: List[string], the malicious passage candidates
        passage_prob_list: List[float], output logits for ground truth answer
    '''
    passages_list, passage_prob_list = [], []
    # calculate original passage logprob
    for passage in ori_passage:
        prompt = '<Question>' + question + '<Background>' + passage + '<Answer>' + answer
        avg_answer_prob = get_model_output_logprobs(prompt=prompt,
                                                    answer=answer)
        passages_list.append(passage)
        passage_prob_list.append(avg_answer_prob)
    # calculate candidate passages logprob
    if updated_passage_list:
        for passage in updated_passage_list:
            prompt = '<Question>' + question + '<Background>' + passage + '<Answer>' + answer
            avg_answer_prob = get_model_output_logprobs(prompt=prompt,
                                                        answer=answer)
            passages_list.append(passage)
            passage_prob_list.append(avg_answer_prob)

    return passages_list, passage_prob_list


def retriever_filter(query, malicious_passages):
    '''one of the similarity filter
    '''
    passages_list = [] # put the original passage in the first place
    for passage in malicious_passages:
        passages_list.append(passage)
    malicious_retriever_score = retriever.retriever_embed(query=query,
                                                passages=passages_list)
    malicious_retriever_score = malicious_retriever_score.detach().cpu().numpy().tolist()

    return malicious_retriever_score


def bert_filter(query, malicious_passages):
    '''one of the similarity filter
    '''
    sentence_bert_score = sentence_bert.calculate_score(query=query,
                                                        passages=malicious_passages)
    sentence_bert_score = sentence_bert_score.detach().cpu().numpy().tolist()
    
    return sentence_bert_score


def rouge2_filter(query, malicious_passages):
    '''one of the similarity filter
    '''
    rouge2_score_value = rouge2_score.calculate_score(query=query, passages=malicious_passages)
    
    return [rouge2_score_value]


def select_new_candidate_passage(malicious_passages, ori_passage_num, malicious_retriever_score, passage_logprob_list):
    '''select the best malicious passage candidate based on similarity score and output logits
    Args:
        malicious passages: List[string], malicious passage candidates
        ori_passage_num: int, the number of original retrieved passages
        malicious_retriever_score: List[float], the similarity score distribution of the malicious passages
        passage_logprob_list: List[float], the gt answer's output logits distribution of the malicious passages
    '''
    candidate_passages_idx = []
    min_retriever_score = min(malicious_retriever_score[:ori_passage_num])
    candidate_logprob_list = passage_logprob_list[ori_passage_num:]
    candidate_retriever_score_list = malicious_retriever_score[ori_passage_num:]
    max_logprob_score = max(passage_logprob_list[:ori_passage_num])
    sorted_logprob_id = sorted(range(len(candidate_logprob_list)), key=lambda k:candidate_logprob_list[k], reverse=False)
    for item in sorted_logprob_id:
        if len(candidate_passages_idx) == ori_passage_num:
            break
        if candidate_logprob_list[item] < max_logprob_score and \
            candidate_retriever_score_list[item] >= min_retriever_score:
            candidate_passages_idx.append(item+ori_passage_num)

    # if there is not enough candidate passages, append the original passages
    if len(candidate_passages_idx) < ori_passage_num:
        print('insufficient qualified candidate passages: ', 
              ori_passage_num - len(candidate_passages_idx))
        for i in range(ori_passage_num-len(candidate_passages_idx)):
            candidate_passages_idx.append(i)
    
    retriever_score_list, logprob_list = [], []
    for idx in candidate_passages_idx:
        retriever_score_list.append(malicious_retriever_score[idx])
        logprob_list.append(passage_logprob_list[idx])

    updated_passages = []
    for idx in candidate_passages_idx:
        updated_passages.append(malicious_passages[idx])

    avg_retriever_score = sum(retriever_score_list)/len(retriever_score_list)
    avg_logprob = sum(logprob_list)/len(logprob_list)

    return updated_passages, avg_retriever_score, avg_logprob


def answer_label(answer):
    '''use NER tool to classify the ground-truth answer in the initialization step
    '''
    answer_stat, annotation_list = {}, []
    if isinstance(answer, list):
        for answer_item in answer:
            sentence = Sentence(answer_item)
            # run NER over sentence
            tagger_pos.predict(sentence)
            tagger_ner.predict(sentence)
            if 'ner' in sentence.annotation_layers:
                annotation_tag = sentence.annotation_layers['ner'][0].data_point.tag
                # annotation_tag = 'NER'
            else:
                # annotations = sentence.annotation_layers['pos']
                annotation_tag = 'POS'
            annotation_list.append(annotation_tag)

        annotation_counts = Counter(annotation_list)
        top_one_tag = annotation_counts.most_common(1)[0]
        if top_one_tag[0] not in answer_stat:
            answer_stat[top_one_tag[0]] = 1
        else:
            answer_stat[top_one_tag[0]] += 1

        answer_label = top_one_tag[0]
    else:
        sentence = Sentence(answer)
        # run NER over sentence
        tagger_pos.predict(sentence)
        tagger_ner.predict(sentence)
        if 'ner' in sentence.annotation_layers:
            annotation_tag = sentence.annotation_layers['ner'][0].data_point.tag
            # annotation_tag = 'NER'
        else:
            # annotations = sentence.annotation_layers['pos']
            annotation_tag = 'POS'

        answer_label = annotation_tag

    return answer_label
    

def target_substution_attack(input_file, pos_substution_rate=0.2, candidates_number=50, setting_type='white-box', filter_type='bert'):
    ''' the main optimization attack function
    Args:
        input_file: string, original/malicious passage data for previous attack results
        pos_substution_rate: float, substution rate for pos type token
        candidates_number: int, number of malicious passage candidate 
        setting_type: string, black-box or white-box
        filter_type: string, the type of similarity filter
    Return:
        optimized malicious passages
    '''
    instances = read_instances(filename=input_file)
    baseline_multiple_passage_results_dict = {}
    cnt = 0

    for key, value in instances.items():
        query = value['question']
        answer = random.choice(value['answer'])
        # labeling answer via entity recognition
        pred_label = answer_label(answer)
        # load data from previous iteration
        backup_tokens = value['top_logtoken']
        original_passages = value['passages']
        baseline_passages = value['malicious_passages']
        updated_passages, original_passages_text, sub_rate_list = [], [], []
        
        for ori_passage, passage, backup_token in zip(original_passages, baseline_passages, backup_tokens):
            if isinstance(ori_passage, dict):
                ori_passage_text = ori_passage['text']
            else:
                ori_passage_text = ori_passage
            original_passages_text.append(ori_passage_text)
            # multi candidates
            for _ in range(candidates_number):
                print('passage length: ', len(passage))
                updated_passage, sub_rate = target_substution_attack_single_sentence(passage=passage,
                                                                        backup_token=backup_token,
                                                                        pred_label=pred_label,
                                                                        pos_substution_rate=pos_substution_rate)
                updated_passages.append(updated_passage)
                sub_rate_list.append(sub_rate)
        print('average substitution rate: ', sum(sub_rate_list)/len(sub_rate_list))
        # calcuate output logits based on llm attacker
        passages, passage_logprob_list = get_pred_logprobs(ori_passage=original_passages_text,
                                                updated_passage_list=updated_passages, 
                                                question=query, answer=answer)
        if setting_type == 'white-box':
            # use the same retriever model as the similarity filter
            malicious_retriever_score = retriever_filter(query=query, malicious_passages=passages)
        else:
            # use an alternative similarity filter
            if filter_type == 'bert':
                malicious_retriever_score = bert_filter(query=query, malicious_passages=passages)
            elif filter_type == 'rouge2':
                malicious_retriever_score = rouge2_filter(query=query, malicious_passages=passages)
            elif filter_type == 'bm25':
                malicious_retriever_score = get_bm25_score(query=query, malicious_passages=passages)
            else:
                raise ValueError('the filter type does not exist')
            
        # select the optimial malicious passage results based on similarity score and output logits
        updated_passages, avg_retriever_score, avg_logprob = select_new_candidate_passage(malicious_passages=passages,
                                                        ori_passage_num=len(original_passages_text),
                                                        malicious_retriever_score=malicious_retriever_score[0],
                                                        passage_logprob_list=passage_logprob_list)

        value['updated_malicious_passages'] = updated_passages
        value['avg_retriever_score'] = avg_retriever_score
        value['avg_logprob'] = avg_logprob

        baseline_multiple_passage_results_dict[key] = value
        cnt += 1
        if len(sub_rate_list) > 0:
            print('overall: ', sum(sub_rate_list)/len(sub_rate_list))
        else:
            print('overall: ', 0)
    
    return baseline_multiple_passage_results_dict


        
if __name__  == '__main__':

    parser = argparse.ArgumentParser(
        description="TPARAG Attack arguments"
    )
    parser.add_argument("--iteraction_number",
                        help="the number of attack iteration",
                        type=int)
    parser.add_argument("--instance_filename",
                        help="the original retrieved passages file path for the given queries",
                        type=str)
    parser.add_argument("--write_filename",
                            help="the result malicious passage file path",
                            type=str)
    
    parser.add_argument("--open_source",
                            help="whether to use an open-source llm attacker",
                            type=bool)
    parser.add_argument("--setting_type",
                            help="black-box setting or white-box attack setting",
                            type=str)
    parser.add_argument("--candidates_number",
                            help="the size of the malicious passage candidate set",
                            type=int)
    parser.add_argument("--c",
                            help="the substution rate of the pos token",
                            type=float)
    
    

    
    args = parser.parse_args()
    write_filename_perfix = args.write_filename.split('.')[0]

    baseline_filename = None
    for i in range(args.iteraction_number):
        write_filename_generate = write_filename_perfix + '_generate_' + str(i) + '.json'
        if i == 0:
            base_filename = args.instance_filename
        else:
            base_filename = baseline_filename
        if os.path.exists(write_filename_generate):
            print('%s has already generated'%write_filename_generate)
        else:
            baseline_results = generate_passage(instance_filename=base_filename,
                                                save_logprob=True, 
                                                open_source=args.open_source)
            write_malicious_passage(instance_list=baseline_results,
                                    filename=write_filename_generate)
            
        write_filename_optimize = write_filename_perfix + '_opt_' + str(i) + '.json'
        if os.path.exists(write_filename_optimize):
            print('%s has already generated'%write_filename_optimize)
        else:
            updated_results = target_substution_attack(input_file=write_filename_generate,
                                                       pos_substution_rate=args.pos_substution_rate,
                                                       setting_type=args.setting_type,
                                                       candidates_number=args.candidates_number)
            if updated_results:
                write_malicious_passage(instance_list=updated_results,
                                        filename=write_filename_optimize)
                print('finished writing substitution')
            else:
                pass
        baseline_filename = write_filename_optimize

    print('Done.')
