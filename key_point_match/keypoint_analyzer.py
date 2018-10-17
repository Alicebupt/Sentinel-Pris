'''
created by xuhong 2018/9/18
'''
import json
import copy
import time
from utils import *


class KeyPointAnalyzer(object):
    def __init__(self, compare_corpus):
        """
        初始化话术分析器，注意从keypoint_analyzer表中读取compare_corpus
        @param compare_corpus: 可参照data/compare_corpus_20.json格式
            {"业务名": 
                {"关键点1":
                    {"compared_corpus": [str],
                     "threshold": {"levenshtein": 0.7,"word2vec": 0.9},
                     "patterns":[str]}
                }
            }
        """
        self.compare_corpus = compare_corpus
        self.compare_corpus_vector = copy.deepcopy(self.compare_corpus)
        for topic in self.compare_corpus_vector:
                for keypoint in self.compare_corpus_vector[topic]:
                    self.compare_corpus_vector[topic][keypoint]["compared_corpus"] = getsimlist_vec(self.compare_corpus[topic][keypoint]["compared_corpus"])
        print('******初始化完成******')

    def get_similarity(self, topic, method, sentence):
        '''
        单句与匹配库匹配,返回高于每项设定的阈值的关键点的最高相似度,有多个关键点的取最高的
        :param topic: string, '现金分期'
        :param method: string, 'levenshtein' or 'word2vec' or 'regex'
        :param sentence: string
        :return result: {'sentence': "subsentence", # 原子句
                         'keypoint':'', 
                         'score':'',  # 相似度分值，regex方法下置为1
                         'compared_source':'匹配库句子', # 匹配库中的句子，regex方法下置为""
                         'regex':''  # regex匹配到的式子
                         }
        '''
        result = []
        keypoints_result = []
        # try:
        #     sim_corpus = self.compare_corpus[topic]
        # except KeyError:
        #     print('******暂不支持该业务分类下的关键点提取！******')
        #     exit()
        sim_corpus = self.compare_corpus[topic]
        for key in sim_corpus.keys():
            if method == 'levenshtein':
                threshold = sim_corpus[key]['threshold']['levenshtein']
                score_result = levenshteinStr(sentence, self.compare_corpus[topic][key]["compared_corpus"], threshold)
                # score_result: 单关键点匹配结果
            elif method == 'word2vec':
                threshold = sim_corpus[key]['threshold']['word2vec']
                score_result = w2v_model_new(sentence, self.compare_corpus_vector[topic][key]["compared_corpus"], threshold)
            elif method == 'regex':
                re_patterns = sim_corpus[key]['patterns']
                score_result = regex(sentence, re_patterns)
            else:
                print('******暂不支持该方法！*******')
            if score_result != None:
                score_result['keypoint'] = key
                score_result['score'] = float('%.2f' % score_result['score']) # 相似度分值取小数点后两位
                keypoints_result.append(score_result)
        # print(keypoints_result)
        if len(keypoints_result) >= 1: # 子句匹配到多个关键点时，取相似度分值最高的
            result = top_keypoint(keypoints_result)
            result['sentence'] = sentence
        else:
            result = None
        return result

    def deal_dialog(self, dialog, topic, N, step):
        '''
        处理输入的一段对话
        只取客服的句子，滑动窗口：对输入字符串按照窗口大小N以步长step取出，返回字符串数组
        :param dialog: list, [{"target": "坐席", "speech": "王先生您好有什么可以帮您", "start_time": "0.00", "end_time": "3.83"},{}]
        :param topic: string, '现金分期'
        :param N: int, 切分句子滑动窗口大小 10
        :param step: int, 滑窗步长 3
        :return subsentence: list, [{'sentence':subsentence1,'sen_num': num, 'start_time':start_time 'end_time':end_time},{'sentence':subsentence2,'sen_num': num,'start_time':start_time 'end_time':end_time}...]
        :return index_sentence: dict, {num: "source sentence"}
        '''
        sentence = []
        subsentence = []
        index_sentence = {}
        if 'sen_num' not in dialog[0].keys():  # 如果给的句子没有标注句子序号，则自动标注，从0开始标号
            for i in range(len(dialog)):
                dialog[i]['sen_num'] = i
        for each_pare in dialog:
            if each_pare['target'] == '坐席':
                string = each_pare['speech']
                sen_num = each_pare["sen_num"]
                index_sentence[sen_num] = string
                if not string:
                    sentence = None
                if N > len(string):
                    subsen = {'sentence': string, 'sen_num': each_pare["sen_num"], 'start_time': each_pare["start_time"],'end_time': each_pare["end_time"]}
                    subsentence.append(subsen)
                else:
                    res = []
                    point = N
                    while point <= len(string):
                        res.append(string[point - N: point])
                        point = point + step
                    for item in res:
                        subsen = {'sentence': item, 'sen_num': each_pare["sen_num"], 'start_time': each_pare["start_time"],'end_time': each_pare["end_time"]}
                        subsentence.append(subsen)
        return subsentence, index_sentence

    def subsenlist_simi(self, subsentence_list, topic, method='levenshtein'):
        '''
        对子句list进行相似度匹配，返回每个子句对应的关键点以及该关键点的相似度分值
        :param subsentence_list: list, [{'sentence':subsentence1,
                                         'sen_num': num, 
                                         'start_time':start_time,
                                         'end_time':end_time},...]
        :param topic: string, '现金分期'
        :param method: string, 用到的算法，默认为levenshtein 可选'word2vec' or 're'
        :return result: list, [{'sentence': '', 
                                'sen_num': 0, 
                                'keypoint': '', 
                                'score': 0.34, 
                                'compared_source': '有什么疑问您可以随时致电我们客服热线。',
                                'regex': '', 
                                'start_time':start_time, 
                                'end_time':end_time}, {}]
        '''
        result = []
        for subsentence in subsentence_list:
            sentence = subsentence['sentence']
            subsentence_result = self.get_similarity(
                topic, method, sentence)
            if subsentence_result != None:
                subsentence_result['sen_num'] = subsentence['sen_num']   # 加入源句，时间等信息
                subsentence_result['start_time'] = subsentence['start_time']
                subsentence_result['end_time'] = subsentence['end_time']
                result.append(subsentence_result)
        if result == []:
            return None
        else:
            return result

    def result_format(self, sentence_result, source_index=None):
        '''
        获取最终结果格式
        :param sentence_result: list, 每一项是单句话返回的结果
                                  [{'sentence': '', 
                                    'sen_num': 0, 
                                    'keypoint': '', 
                                    'score': 0.34, 
                                    'compared_source': '有什么疑问您可以随时致电我们客服热线。',
                                    'regex': '', 
                                    'start_time':start_time, 
                                    'end_time':end_time}, {}]
        :param source_index: dict, 句子index，用于获取子句对应的原句
        :return result: list, [{'keypoint': '11.确认卡片是否在手', 
                                'matched': [{'sentence': '',  # 原句或子句
                                             'score': 0.53,   # 相似度分值
                                             'compared_source': '', # 匹配库中的句子
                                             'regex': '', # 匹配的正则表达式
                                             'start_time': '0.00', 
                                             'end_time': '3.83', 
                                             'source_sentence': ''  # 子句源句
                                            }]
                                },,,] '''
        result = []
        keypoint_list = []
        if sentence_result == None:
            result = []
            return result
        for subsentence in sentence_result:
            keypoint,sentence_num = subsentence['keypoint'],subsentence['sen_num']
            source_sentence = source_index[sentence_num]
            subsentence.pop('keypoint')
            subsentence.pop('sen_num')
            subsentence['source_sentence'] = source_sentence
            if keypoint not in keypoint_list:
                keypoint_list.append(keypoint)
                result.append({'keypoint': keypoint, 'matched': []})
            result[keypoint_list.index(
                keypoint)]['matched'].append(subsentence)
        if result == []:
            return None
        else:
            return result

    def run_levenshtein(self, transcripts, topic):
        '''
        对某个业务分类下的单个对话，获取关键点及对应的句子.如果没有检测到任何关键点，返回一个空[]
        :param  transcripts: list,每一项为一个句子
                        示例：[
                                {
                                    "target": "坐席",
                                    "speech": "王先 生您好有什么可以帮您",
                                    "start_time": "0.00",
                                    "end_time": "3.83"
                                },,,,
                            ]
        :param  topic:  string，业务分类，示例：'现金分期'
        :return result: list, 每一项为这段话匹配到的关键点之一，
                        格式： [{'keypoint': '11.确认卡片是否在手', 
                                'matched': [{'sentence': '',  # 匹配到的原句中的句子
                                             'score': 0.53,   # 相似度分值
                                             'compared_source': '', # 匹配库中的句子
                                             'regex': '', # 置空
                                             'start_time': '0.00', 
                                             'end_time': '3.83', 
                                             'source_sentence': ''  # 子句源句
                                            }]
                                },,,] 
        '''
        subsentence_list, index_sentence = self.deal_dialog(
            transcripts, topic, 10, 3)  # 分句 滑窗大小N=10，滑窗步长step=3
        dialog_result = self.subsenlist_simi(
            subsentence_list, topic, method="levenshtein")  # 对分句结果list获取匹配结果
        result = self.result_format(
            sentence_result=dialog_result, source_index=index_sentence)
        if result != []: # 同一源句合并
            for index in range(len(result)):
                matched = result[index]['matched']
                result[index]['matched'] = combine(matched)
        return result

    def run_word2vec(self, transcripts, topic):
        '''
        对某个业务分类下的单个对话，获取关键点及对应的句子.如果没有检测到任何关键点，返回一个空[]
        :param  transcripts: list,每一项为一个句子
                        示例：[
                                {
                                    "target": "坐席",
                                    "speech": "王先 生您好有什么可以帮您",
                                    "start_time": "0.00",
                                    "end_time": "3.83"
                                },,,,
                            ]
        :param  topic:  string，业务分类，示例：'现金分期'
        :return result: list, 每一项为这段话匹配到的关键点之一，
                        格式：[
                                {'keypoint':'',
                                'matched':[{'sentence':'',  # 切割后的句子
                                            'compared_source':'',  # 匹配库中的句子
                                            'source_sentence':'',  # 对话中的原句（未切割）
                                            'score': float,
                                            "start_time": "08:06:38",
                                            "end_time": "08:06:43",
                                            "regex": str
                                            },
                                            {},,,
                                          ]
                                },
                                {},
                            ]
        '''
        # subsentence_list, index_sentence = self.deal_dialog(
        #     dialog, topic, 10, 3)  # 分句 滑窗大小N=10，滑窗步长step=3
        # [{'sentence':subsentence1,'sen_num': num, 'start_time': str, "end_time": str}]
        subsentence_list = []
        index_sentence = {}  # {num: "source sentence"}
        for i, item in enumerate(transcripts):
            subsentence_list.append({
                "sentence": item["speech"],
                'sen_num': i,
                "start_time": item['start_time'],
                "end_time": item['end_time']
            })
            index_sentence[i] = item["speech"]
        dialog_result = self.subsenlist_simi(
            subsentence_list, topic, method="word2vec")  # 对分句结果list获取匹配结果
        result = self.result_format(
            sentence_result=dialog_result, source_index=index_sentence)
        return result

    def run_regex(self, transcripts, topic):
        '''
        对某个业务分类下的单个对话，获取关键点及对应的句子
        :param  transcripts: list,每一项为一个句子
                        示例：[
                                {
                                    "target": "坐席",
                                    "speech": "王先 生您好有什么可以帮您",
                                    "start_time": "0.00",
                                    "end_time": "3.83"
                                },,,,
                            ]
        :param  topic:  string，业务分类，示例：'现金分期'
        :return result: list, 每一项为这段话匹配到的关键点之一，
                        格式：[
                                {'keypoint':'',
                                'matched':[{'sentence':'',  # 切割后的句子
                                            'compared_source':'',  # 空
                                            'source_sentence':'',  # 对话中的原句（未切割）
                                            'score': float,
                                            "start_time": "08:06:38",
                                            "end_time": "08:06:43",
                                            "regex": str
                                            },
                                            {},,,
                                        ]
                                },
                                {},
                            ]
        '''

        # subsentence_list, index_sentence = self.deal_dialog(
        #     dialog, topic, 10, 3)  # 分句 滑窗大小N=10，滑窗步长step=3
        # [{'sentence':subsentence1,'sen_num': num, 'start_time': str, "end_time": str}]
        subsentence_list = []
        index_sentence = {}  # {num: "source sentence"}
        for i, item in enumerate(transcripts):
            subsentence_list.append({
                "sentence": item["speech"],
                'sen_num': i,
                "start_time": item['start_time'],
                "end_time": item['end_time']
            })
            index_sentence[i] = item["speech"]
        dialog_result = self.subsenlist_simi(
            subsentence_list, topic, method="regex")  # 对分句结果list获取匹配结果
        result = self.result_format(
            sentence_result=dialog_result, source_index=index_sentence)
        return result

    def transform_result_list_to_dict(self, from_format):
        '''
        转换格式，将结果的list格式转换为字典格式
        @param from_format： list, 每一项为这段话匹配到的关键点之一，
                            格式： [{'keypoint': '11.确认卡片是否在手', 
                                    'matched': [{'sentence': '',  # 匹配到的原句中的句子
                                                 'score': 0.53,   # 相似度分值
                                                 'compared_source': '', # 匹配库中的句子
                                                 'regex': '', # 置空
                                                 'start_time': '0.00', 
                                                 'end_time': '3.83', 
                                                 'source_sentence': ''  # 子句源句
                                                }]
                                    },,,] 
        @return to_format: dict,{
                                '11.确认卡片是否在手': {'keypoint': keypoint, # 所属关键点
                                                     'sentence': '',  # 匹配到的原句中的句子
                                                     'score': 0.53,   # 相似度分值
                                                     'compared_source': '', # 匹配库中的句子
                                                     'regex': '', # 置空
                                                     'start_time': '0.00', 
                                                     'end_time': '3.83', 
                                                     'source_sentence': ''  # 子句源句},
                                             
                                            }],,,
                                    } 
        '''
        to_format={}
        if from_format == []:
            return to_format
        for item in from_format:
            for one_sentence in item['matched']:
                to_format[one_sentence['sentence']] = one_sentence
                to_format[one_sentence['sentence']]['keypoint'] = item['keypoint']
        return to_format

    def transform_result_dict_to_list(self, from_format):
    	'''
    	转换格式，将结果的dict格式转换为list格式
    	@param from_format： dict,{
                                '11.确认卡片是否在手': {'keypoint': keypoint, # 所属关键点
                                                     'sentence': '',  # 匹配到的原句中的句子
                                                     'score': 0.53,   # 相似度分值
                                                     'compared_source': '', # 匹配库中的句子
                                                     'regex': '', # 置空
                                                     'start_time': '0.00', 
                                                     'end_time': '3.83', 
                                                     'source_sentence': ''  # 子句源句},
                                             
                                            }],,,
                                    } 
	    @return to_format:  list, 每一项为这段话匹配到的关键点之一，
	                        格式： [{'keypoint': '11.确认卡片是否在手', 
	                                'matched': [{'sentence': '',  # 匹配到的原句中的句子
	                                             'score': 0.53,   # 相似度分值
	                                             'compared_source': '', # 匹配库中的句子
	                                             'regex': '', # 置空
	                                             'start_time': '0.00', 
	                                             'end_time': '3.83', 
	                                             'source_sentence': ''  # 子句源句
	                                            }]
	                                },,,] 
    	'''
    	to_format = []
    	for sentence in from_format.keys():
            flag = 0 # 句子内容是否被成功添加至结果里的标志
            for keypoint_index in range(len(to_format)):
                keypoint = to_format[keypoint_index]['keypoint']
                if from_format[sentence]['keypoint'] == keypoint:
                    del from_format[sentence]['keypoint']
                    to_format[keypoint_index]['matched'].append(from_format[sentence])
                    flag = 1
            if flag == 0: # 未匹配到已有关键点，则新增关键点
                new_keypoint = {'keypoint':from_format[sentence]['keypoint']}
                del from_format[sentence]['keypoint']
                new_keypoint['matched'] = [from_format[sentence]]
                to_format.append(new_keypoint)
    	return to_format


    def combine_result(self, le_result=[], regex_result=[], word2vec_result=[]):
        '''
    	合并一个对话通过不同算法获得的结果，合并优先级看score，regex的score为1
    	@param le_result: dict, levenshtein算法获得的结果
    	       regex_result： dict, 正则表达式获得的结果
    	       word2vec_result： dict, word2vec算法获得的结果
    	@return combine_result: list, 合并之后的结果


        dict,{
            '11.确认卡片是否在手': {'keypoint': keypoint, # 所属关键点
                                 'sentence': '',  # 匹配到的原句中的句子
                                 'score': 0.53,   # 相似度分值
                                 'compared_source': '', # 匹配库中的句子
                                 'regex': '', # 置空
                                 'start_time': '0.00', 
                                 'end_time': '3.83', 
                                 'source_sentence': ''  # 子句源句},
                         
                        }],,,
                } 
    	'''
        le_result_dic = self.transform_result_list_to_dict(le_result) # 将获得的结果的list格式转化为dict格式方便后续处理
        # print(le_result_dic)
        regex_result_dic = self.transform_result_list_to_dict(regex_result)
        # print(regex_result_dic)
        word2vec_result_dic = self.transform_result_list_to_dict(word2vec_result)
        combine_result_dict = {}
        all_result = [regex_result_dic, word2vec_result_dic, le_result_dic] # 有一定的优先级，当分数相同时取前面的结果
        for result in all_result:  # 对所有结果进行遍历，对于同一句话，按相似度分值判断取哪个方法的结果
            if result == {}:
                continue
            for sentence in result.keys():
                if sentence not in combine_result_dict.keys():
                    combine_result_dict[sentence] = result[sentence]
                else:
                    new_score = result[sentence]['score']
                    last_score = combine_result_dict[sentence]['score']
                    if new_score > last_score: # 只有分数值大于前面的结果时才能覆盖，即分数相等时，前面的结果优先级更高
                        combine_result_dict[sentence] = result[sentence]
        print(combine_result_dict)
        combine_result = self.transform_result_dict_to_list(combine_result_dict)
        return combine_result


    def test(self, dialogs):
        '''
        测试多个对话 
        @param dialogs : [{"transcripts": [{},{}], "id": str, "topic":str}, {"transcripts": [{},{}], "id": str,"topic":str}]
        @return 只返回matched到的对话,[{
            "id": str,
            "matched": [
                            {'keypoint':'',
                            'matched':[{'sentence':'',  # 切割后的句子
                                        'compared_source':'',  # 匹配库中的句子
                                        'source_sentence':'',  # 对话中的原句（未切割）
                                        'score': float,
                                        "regex": str,      
                                        "start_time": "08:06:38",
                                        "end_time": "08:06:43",
                                        },
                                        {},,,
                                ]
                            ]},
                            {},
                        ],
            "transcripts": str(dumped)
            }]
        '''
        matched = []
        for dialog in dialogs:
            id = dialog["id"]
            transcripts = dialog["transcripts"]
            topic = dialog["topic"]
            if topic not in self.compare_corpus.keys():
            	return '暂不支持该业务分类下的关键点提取！'
            le_result = self.run_levenshtein(transcripts=transcripts, topic=topic)
            regex_result = self.run_regex(transcripts=transcripts, topic=topic)
            w2v_result = self.run_word2vec(transcripts=transcripts, topic=topic)
            print(le_result)
            print(regex_result)
            print(w2v_result)

            # 合并算法结果
            result = self.combine_result(le_result=le_result, regex_result=regex_result,word2vec_result=w2v_result)
            
            if result != []:
                matched.append({
                    "id": id,
                    "matched": result,
                    "transcripts": json.dumps(transcripts, ensure_ascii=False)
                })
        return matched

if __name__ == '__main__':
    # 加载匹配库
    with open('data/compare_corpus_20.json', 'r', encoding='utf8') as f:
        compare_corpus = json.load(f)

    # key_point = KeyPointAnalyzer(compare_corpus={
    #     "现金分期": {
    #         "确认业务": {
    #             "compared_corpus": [
    #                 "您要办理新快现业务？",
    #                 "您要现金是么？",
    #                 "您说的是现金分期业务么？",
    #                 "您是要把现金转账到储蓄卡的业务吗？",
    #                 "您说的是新快现业务。",
    #                 "是转现金给您么？"
    #             ],
    #             "threshold": {
    #                 "levenshtein": 0.2,
    #                 "word2vec": 0.8
    #             },
    #             "patterns": ["资金用途"]
    #         }}})
    key_point = KeyPointAnalyzer(compare_corpus=compare_corpus)
    transcripts = [
        {
            "target": "坐席",
            "speech": "好谢谢您那现在麻烦您把信用卡翻到背面卡片在手上的是吧请您把信用卡翻到背面白色签名条上有七位数字给您一个语音提示请您把七位数字中的后三位输入进来验证一下好吧",
            "start_time": "0.00",
            "end_time": "3.83"
        },
        {
            "target": "客户",
            "speech": "你好你帮我查一下我这个信用卡这个月我明明都已经还完了怎么怎么又差了一万多回去了什么意思啊",
            "start_time": "3.83",
            "end_time": "17.86"
        },
        {
            "target": "坐席",
            "speech": "请问您的使用资金用途是干嘛？",
            "start_time": "17.86",
            "end_time": "23.92"
        },
        {
            "target": "客户",
            "speech": "请问你的储蓄卡是一类账户么？本期账单还款还需要",
            "start_time": "23.92",
            "end_time": "30.94"
        },
        {
            "target": "坐席",
            "speech": "呃稍等一下就是您信用卡本期账单还款还需要再还一万一一送一还了一万元了",
            "start_time": "30.94",
            "end_time": "41.79"
        },
        {
            "target": "客户",
            "speech": "我总共这个额度我还三万八是吗",
            "start_time": "41.79",
            "end_time": "46.25"
        },
        {
            "target": "坐席",
            "speech": "请王先生具体记录吧把您信用卡的密码输入验证一下我看一下您能还记录好吗",
            "start_time": "46.25",
            "end_time": "57.10"
        },
        {
            "target": "客户",
            "speech": "你们就剩一点",
            "start_time": "57.10",
            "end_time": "59.01"
        },
        {
            "target": "坐席",
            "speech": "好的那这边和验证通过了那您这张中信信用卡的最后一笔交易是在是在什么时候或者是多少金额您要提供一下",
            "start_time": "59.01",
            "end_time": "64.44"
        },
        {
            "target": "客户",
            "speech": "aa",
            "start_time": "64.44",
            "end_time": "65.07"
        },
        {
            "target": "坐席",
            "speech": "您的账单的话是三万八千六百五十九块六毛七之前还了有两万七千六百五十九块六毛七了然后的话呢在二十八二十七号二十三点",
            "start_time": "65.07",
            "end_time": "82.94"
        },
        {
            "target": "客户",
            "speech": "我想换张卡",
            "start_time": "82.94",
            "end_time": "84.53"
        },
        {
            "target": "坐席",
            "speech": "是还了一个一万一没错这个款项呢是结算的话呢由于只是原因的话的话到二十八号的一个这边的话呢也算入账所以您这边的话账单里面显示你没有还清但实际上的话您是已经还清了不用再还了三十一号您放心",
            "start_time": "84.53",
            "end_time": "113.56"
        },
        {
            "target": "客户",
            "speech": "唉你们这个什么这样查都看的我那我得干的查了好一阵花了它它不有一个叫是拿我的倒数对答都我都明明还清了怎么还",
            "start_time": "113.56",
            "end_time": "130.15"
        },
        {
            "target": "坐席",
            "speech": "给您添麻烦了非常抱歉因为您这个是还款时间的话呢比较晚的一个情况导致这边的话呢一个入账时间的话呢也可以给您添麻烦了",
            "start_time": "130.15",
            "end_time": "148.01"
        },
        {
            "target": "客户",
            "speech": "我刚才查了刚才显示的要还款一万多一万吧",
            "start_time": "148.01",
            "end_time": "154.07"
        },
        {
            "target": "客户",
            "speech": "噢那你们的系统怎么都没了消费下班了呢",
            "start_time": "167.79",
            "end_time": "173.53"
        },
        {
            "target": "坐席",
            "speech": "给您添麻烦了非常抱歉",
            "start_time": "173.53",
            "end_time": "176.72"
        }
    ]
    topic = '现金分期'
    # t = time.time()
    # for i in range(1):
    #     result = key_point.run_levenshtein(transcripts=transcripts, topic=topic)
    #     print("测试单个对话:", result)
    # for i in range(1):
    #     result = key_point.run_word2vec(transcripts=transcripts, topic=topic)
    #     print("测试单个对话:", result)
    # # print(time.time()-t)
    # exit()

    # test dialogs,后台实际调用的function
    t1 = time.time()
    dialogs = [{"transcripts": transcripts, "id": "dfrvfv", "topic":topic}]
    print("测试多个对话：", key_point.test(dialogs=dialogs), time.time()-t1)
    # le_result = [{'keypoint': '确认资金用途', 'matched': [{'sentence': '请问您资金用途是？', 'score': 0.7, 'compared_source': '请问您资金用途是？', 'regex': '', 'start_time': '17.86', 'end_time': '23.92', 'source_sentence': '请问您资金用途是？'}]}]
    # regex_result = [{'keypoint': '确认资金用途', 'matched': [{'sentence': '请问您资金用途是？', 'score': 1.0, 'compared_source': '请问您资金用途是？', 'regex': 'ssssss', 'start_time': '17.86', 'end_time': '23.92', 'source_sentence': '请问您资金用途是？'}]},{'keypoint': '转IVR确认储蓄卡卡号', 'matched': [{'sentence': '好谢谢您那现在麻烦您把信用卡翻到背面卡片在手上的是吧请您把信用卡翻到背面白色签名条上有七位数字给您一个语音提示请您把七位数字中的后三位输入进来验证一下好吧', 'score': 1.0, 'compared_source': '', 'regex': '语音提示', 'start_time': '0.00', 'end_time': '3.83', 'source_sentence': '好谢谢您那现在麻烦您把信用卡翻到背面卡片在手上的是吧请您把信用卡翻到背面白色签名条上有七位数字给您一个语音提示请您把七位数字中的后三位输入进来验证一下好吧'}]}, {'keypoint': '确认储蓄卡是否为I类账户', 'matched': [{'sentence': '请问你的储蓄卡是一类账户么？本期账单还款还需要', 'score': 1.0, 'compared_source': '', 'regex': '是(.*)一类账户(吗|么|嘛)', 'start_time': '23.92', 'end_time': '30.94', 'source_sentence': '请问你的储蓄卡是一类账户么？本期账单还款还需要'}]}, {'keypoint': '确认储蓄卡开户6个月内是否有交易', 'matched': [{'sentence': '好的那这边和验证通过了那您这张中信信用卡的最后一笔交易是在是在什么时候或者是多少金额您要提供一下', 'score': 1.0, 'compared_source': '', 'regex': '最后.?一笔', 'start_time': '59.01', 'end_time': '64.44', 'source_sentence': '好的那这边和验证通过了那您这张中信信用卡的最后一笔交易是在是在什么时候或者是多少金额您要提供一下'}]}]
    # w2v_result = [{'keypoint': '确认资金用途', 'matched': [{'sentence': '请问您资金用途是？', 'score':0.8, 'compared_source': '请问您资金用途是？', 'regex': '', 'start_time': '17.86', 'end_time': '23.92', 'source_sentence': '请问您资金用途是？'}]},{'keypoint': '转IVR确认储蓄卡卡号', 'matched': [{'sentence': '好谢谢您那现在麻烦您把信用卡翻到背面卡片在手上的是吧请您把信用卡翻到背面白色签名条上有七位数字给您一个语音提示请您把七位数字中的后三位输入进来验证一下好吧', 'score': 1.0, 'compared_source': '', 'regex': '语音提示', 'start_time': '0.00', 'end_time': '3.83', 'source_sentence': '好谢谢您那现在麻烦您把信用卡翻到背面卡片在手上的是吧请您把信用卡翻到背面白色签名条上有七位数字给您一个语音提示请您把七位数字中的后三位输入进来验证一下好吧'}]}, {'keypoint': '确认储蓄卡是否为I类账户', 'matched': [{'sentence': '请问你的储蓄卡是一类账户么？本期账单还款还需要', 'score': 1.0, 'compared_source': '', 'regex': '是(.*)一类账户(吗|么|嘛)', 'start_time': '23.92', 'end_time': '30.94', 'source_sentence': '请问你的储蓄卡是一类账户么？本期账单还款还需要'}]}, {'keypoint': '确认储蓄卡开户6个月内是否有交易', 'matched': [{'sentence': '好的那这边和验证通过了那您这张中信信用卡的最后一笔交易是在是在什么时候或者是多少金额您要提供一下', 'score': 1.0, 'compared_source': '', 'regex': '最后.?一笔', 'start_time': '59.01', 'end_time': '64.44', 'source_sentence': '好的那这边和验证通过了那您这张中信信用卡的最后一笔交易是在是在什么时候或者是多少金额您要提供一下'}]}]
    # print("测试算法融合结果：", key_point.combine_result(le_result=le_result))
