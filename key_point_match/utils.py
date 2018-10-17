import json,copy,re
import Levenshtein
import linecache
from gensim.models import Word2Vec
import numpy as np
import jieba
from scipy.spatial.distance import pdist

stopwords_path = "data/stopwords.txt"
model_path= "model/word2vec_include.model"
model_loaded= Word2Vec.load(model_path)
wordvec_size = len(model_loaded['账单'])
zero_pad = [0 for n in range(wordvec_size)]
stopwordlist = [line.strip() for line in open(stopwords_path, 'r', encoding="utf-8")]


def corpus(path):
    '''
    读取匹配库
    @param path: './corpus_19.json'
    @return: {"申请办卡"：["要办卡"]}
    '''
    with open(path, 'r', encoding="utf-8")as f:
        corpus = json.load(f)
    return corpus

def data(path):
    '''
    读取过滤后的数据
    @param path: 数据的路径:'./data_final.json'
    @return: [["想办卡","好的"], []]
    '''
    with open(path, 'r')as f:
        data = json.load(f)
    new_text = []
    for eachdialog in data.items():
        new_text.append(eachdialog[1])
    return new_text

def sentenceSplit(string, N, step):
    """
    滑动窗口：对输入字符串按照窗口大小N以步长step取出，返回字符串数组
    @param  string<string>:查一下信用卡额度
    @param  N<int>：5
    @param  step<int>：1
    @return ["查一下信用卡","一下信用卡额","下信用卡额度"]
    """
    if not string:
        return []
    if N > len(string):
        return [string]
    if N == 1:
        return string

    res = []
    point = N
    while point <= len(string):
        res.append(string[point - N: point])
        point = point + step
    return res

def levenshteinStr(sentence, simlist, threshold):
    """
    单句与匹配句子list做相似度计算，返回相似度分值最高的一个
    @param sentence:string, 原句
    @param simlist:list, 匹配句子list
    @param threshold:float, 阈值
    @return {'sentence': "subsentence", # 原子句
             'score':'',  # 相似度分值，regex方法下置为1
             'compared_source':'匹配库句子', # 匹配库中的句子，regex方法下置为""
             'regex':''  # regex匹配到的式子
            }
    """
    sim_temp_dict = {}
    sim_temp = float(threshold)
    for eachsim in simlist:
        score = Levenshtein.ratio(eachsim, sentence)
        if score > sim_temp:
            sim_temp = score
            sim_temp_dict = {'sentence': sentence, # 原子句
                             'score':sim_temp,  # 相似度分值
                             'compared_source':eachsim, # 匹配库中的句子
                             'regex':''}
        else:
            continue
    if not sim_temp_dict:
        return None
    else:
        return sim_temp_dict

def top_keypoint(keypoints):
    '''
    取多个关键点的top1：
    @param keypoints: list,[{'keypoint':'key1', 'score':0.2, 'compared_source':'sdd'}, {'keypoint':'key2', 'score':0.9, 'compared_source':'top1_compared_sentence'}, {'keypoint':'key3', 'score':0.6, 'compared_source':'top1_compared_sentence1'}]
    @return top1_keypoint:{'compared_source': 'top1_compared_sentence', 'keypoint': 'key2', 'score': 0.9}
    '''
    keypoint_list = [item['keypoint'] for item in keypoints]
    score = [item['score'] for item in keypoints]
    score_forindex = copy.deepcopy(score)
    score.sort()
    index = score_forindex.index(score[-1])
    top1_keypoint = keypoints[index]
    return top1_keypoint

def w2v_model_new(sentence, simi_list, threshold):
    """
    @param sentence:str
    @param simi_list:  [{"sentence":str, "array":array}]
    @return sim_temp_dict:{'sentence':  '', # 原子句
                           'score':sim_temp,  # 相似度分值
                           'compared_source':'', # 匹配库中的句子
                           'regex':'' # 置为空}
    """
    sentence_vec=get_vec(sentence)
    sim_temp_dict = {}
    sim_temp = float(threshold)
    #print('simi',simi_list)
    for eachsim in simi_list:
        # print("eachsim", eachsim)
        score = getscore(sentence_vec[1], eachsim['array'])
        if score > sim_temp:
            sim_temp = score
            sim_temp_dict = {'sentence':  get_vec(sentence)[0], # 原子句
                             'score':sim_temp,  # 相似度分值
                             'compared_source':eachsim['sentence'], # 匹配库中的句子
                             'regex':''}
        else:
            continue
    if not sim_temp_dict:
        return None
    else:
        return sim_temp_dict


def getsimlist_vec(list):
    """
    匹配库向量化
    @param list: ["", ""]
    @return: [{"sentence": str, "array": np.array([])}]
    """
    result = []
    for eachsentence in list:
        wordlist = jieba.cut(eachsentence)
        wordlist = [word for word in wordlist if word not in stopwordlist]
        count = 0
        sum = zero_pad
        for eachword in wordlist:
            try:
                temp = model_loaded[eachword]
                count = count + 1
            except KeyError:
                temp = zero_pad
            sum = np.array(sum) + np.array(temp)
        if count == 0:
            continue
        result.append({"sentence": eachsentence, "array": sum / count})
    return result
    
def getscore(sentence_vec, sim_vec):
    """
    计算两个向量的余弦相似度
    @param sentence_vec: np.array([]))
    @param sim_vec: np.array([]))
    @return: score
    """
    if list(sentence_vec) == list(zero_pad):
        score = 1
    else:
        score = pdist(np.vstack([sentence_vec, sim_vec]), 'cosine')
    return 1 - score


def get_vec(sentence):
    s = jieba.cut(sentence)
    s=[word for word in s if word not in stopwordlist] 
    count = 0
    sum = zero_pad

    for eachword in s:
        try:
            temp = model_loaded[eachword]
            count = count + 1
        except KeyError:
            temp = zero_pad
        sum = np.array(sum) + np.array(temp)
    x = {}
    if count == 0:
        x[0] = sentence
        x[1] = zero_pad
        return x
    else:
        x[0] = sentence
        x[1] = sum / count
        return x

def regex(sentence, keywords):
    """
    单句与keywords做正则匹配，包含keywords里的任意一个则匹配成功
    @param sentence:string, 原句
    @param keywords:list, keywords list
    @return {'sentence': sentence, # 原子句
             'score':1,  # 相似度分值
             'compared_source':'',
             'regex':pattern}
    """
    matched_list = []
    if keywords == []:
        return None
    else:
        for pattern in keywords:
            if re.search(pattern, sentence):
                return {'sentence': sentence, # 原子句
                        'score':1,  # 相似度分值
                        'compared_source':'',
                        'regex':pattern}
        return None

def combine(matched):
    """
    对一个关键点下匹配到的多个句子进行筛选，来自于同一源句的取分值最高的一个
    @param matched:list,  [{'sentence': '',  # 匹配到的原句中的句子
                            'score': 0.53,   # 相似度分值
                            'compared_source': '', # 匹配库中的句子
                            'regex': '', # 置空
                            'start_time': '0.00', 
                            'end_time': '3.83', 
                            'source_sentence': ''  # 子句源句
                           },,,,] 
    @return list,   [{'sentence': '',  # 匹配到的原句中的句子
                      'score': 0.53,   # 相似度分值
                      'compared_source': '', # 匹配库中的句子
                      'regex': '', # 置空
                      'start_time': '0.00', 
                      'end_time': '3.83', 
                      'source_sentence': ''  # 子句源句
                    },,,,] 
    """
    score = [item['score'] for item in matched]
    score_forindex = copy.deepcopy(score)
    score.sort()
    source_sentence_list = []
    result = []
    for i in range(len(score)):
        index = score_forindex.index(score[len(score)-i-1])
        if matched[index]['source_sentence'] not in source_sentence_list:
            result.append(matched[index])
            source_sentence_list.append(matched[index]['source_sentence'])
            matched.remove(matched[index])
            score_forindex.remove(score[len(score)-i-1])    
    return result
