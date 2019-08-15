# -*- coding: utf-8 -*-
# Date    : 2019-08-15 21:58:57
# Author  : KunYang

import re
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
!pip install apyori
import sys
sys.path.append("/opt/conda/lib/python3.5/site-packages")
from apyori import apriori

# 一些宏定义
punctuation_list = "[,.!]"
stop_words = set(stopwords.words("english"))
for i in ['thing', 'something', 'anything', 'nothing', 'everything']:
    stop_words.add(i)
with open('/home/kesci/input/feature8835/positive-words.txt') as f:
    positive_words = set(f.read().split('\r\n'))
with open('/home/kesci/input/feature8835/negative-words.txt') as f:
    negative_words = set(re.sub('\r','',f.read()).split('\n'))
 
def del_punctuation(sts):
    '''删除标点
    输入：句子；返回值：去除标点后的单词列表'''
    return re.sub(punctuation_list, "", sts).split()

def del_stopwords(sts_list):
    '''删停用词
    输入：单词列表；返回值：删除停用词后的单词构成的列表'''
    return [word for word in sts_list if word.lower() not in stop_words]

def get_feature_and_text(review):
    '''获取特征及对应句子
    输入：一段评论；返回值：该段评论中包含的feature(s)及其对应句子'''
    review_features = []
    review_texts = []
    for line in review.split("\r\n"):
        if "##" in line:
            feat_text = line.split('##')
            if feat_text[0]:
                review_features.append(feat_text[0])
                review_texts.append(feat_text[1])
    return review_features, review_texts

def lemmatize_text(sent_list, pos_list):
    '''词形还原
    sent_list：单词构成的列表
    pos_list：要处理的词性列表，比如['a','v','n]
    返回值：还原后的单词列表'''
    lem = WordNetLemmatizer()
    lem_list = sent_list[:]
    for p in pos_list:
        lem_list = [lem.lemmatize(w, pos=p).encode('utf-8') for w in lem_list]
    return lem_list

def pos_tagging(sts_list):
    '''词性标注
    输入：单词构成的列表；返回值：词性标注的结果'''
    sentence_segs = word_tokenize(" ".join(sts_list))
    return pos_tag(sentence_segs)

def find_phrase(pos_list):
    pos_result = [w for w in pos_list if w[0].lower() not in stop_words]
    phrases = []
    grammer = ['NP: {<JJ><NN>}','NP:{<NN><NN>}']
    for g in grammer:
        rp = RegexpParser(g)
        result = rp.parse(pos_result)
        for i in result:
            if isinstance(i, nltk.tree.Tree):
                phrases.append(i[0][0]+' '+i[1][0])
    return phrases
    
def get_freq_nouns(list_data,sp):
    '''找到频繁的名词'''
    freq_nouns = []
    apr = apriori(list_data, min_support=sp, max_length=1)
    for result_apri in apr:
        freq_nouns.extend([x for x in result_apri.items])
    return freq_nouns
    
def get_freq_phrases(list_data,sp):
    '''找到频繁的词组'''
    freq_phrases = []
    apr = apriori(list_data, min_support=sp, max_length=1)
    for result_apri in apr:
        freq_phrases.extend([x for x in result_apri.items])
    return freq_phrases

def get_info_from_reviews(rvs):
    '''输入：reviews
        返回值：(all_nouns,all_phrases,real_features,individual_rv_len)'''
    pos_results,all_nouns,all_phrases,adjs,real_features = [],[],[],[],[] # 整个文档的信息
    individual_rv_len = []
    for review in rvs:
        # 找到人工标注的features
        features, review_texts = get_feature_and_text(review)
        individual_rv_len.append(len(review_texts))
        real_features.extend(features)
        for sentence in review_texts:
            # 删除标点
            without_punc = del_punctuation(sentence)
            # 删停用词
            without_stop = del_stopwords(without_punc)
            # 词形还原
            lem_result1 = lemmatize_text(without_punc, ['a','n','v'])
            lem_result2 = lemmatize_text(without_stop, ['a','n','v'])
            # 词性标注
            pos_result1 = pos_tagging(lem_result1)
            pos_result2 = pos_tagging(lem_result2)
            
            noun_list = [w[0] for w in pos_result2 if w[1] == 'NN' or w[1] == 'NNS']
            adj_list = [w[0] for w in pos_result1 if w[1]=='JJ' or w[1]=='RB']
            adj_list = [i for i in adj_list if i not in stop_words]
            # nouns_idx = [lem_result1.index(w) for w in [noun_list]]
            # real_adjs = []
            # for w in adj_list:
            #     idx =  lem_result1.index(w)
            #     for i in nouns_idx:
            #         if abs(idx-i)<5:
            #             real_adjs.append(w)
            #             break
            # 词组
            phrases = find_phrase(pos_result2)
            
            pos_results.append(pos_result1)
            all_nouns.append(noun_list)
            all_phrases.append(phrases)
            adjs.append(adj_list)
    return pos_results,all_nouns,all_phrases,adjs,real_features,individual_rv_len




import numpy as np

for text in ['Canon','DVD','Nikon','Nokia6610','Nomad']:
    with open("/home/kesci/input/feature8835/%s.txt" %text)as f:
        data = f.read()
    all_reviews = data.split("[t]")[1:]
    pos_results,all_nouns,all_phrases,adjs,real_features,indiv_rv_len = \
                                get_info_from_reviews(all_reviews)
    
    indiv_rv_len = np.cumsum(indiv_rv_len).tolist()
    def get_group(idx):
        for i in range(len(indiv_rv_len)):
            if idx < indiv_rv_len[i]:
                return i
                break
    
    freq_nouns_in_reviews = get_freq_nouns(all_nouns,sp=0.01) # sp:最小支持度
    freq_phrases_in_reviews = get_freq_phrases(all_phrases,sp=2./len(all_nouns))
    freq_phrases_in_reviews_set = [set(i.split()) for i in freq_phrases_in_reviews]
    
    sentences_features = [] # 每一句话包含的features
    superset_index = set() # 存在超集的句子索引
    for c_idx,(nouns,phrases) in enumerate(zip(all_nouns,all_phrases)):
        nouns = set(nouns)
        phrases = set(phrases)
        feat = []
        feat_phra = []
        for n in nouns:
            if n in freq_nouns_in_reviews:
                feat.append(n)
        for p in phrases:
            p = set(p.split()) 
            if p in freq_phrases_in_reviews_set:
                if p&set(feat)==p:
                    superset_index.add(c_idx)
                idx = freq_phrases_in_reviews_set.index(p)
                feat_phra.append(freq_phrases_in_reviews[idx])
            
        sentences_features.append(feat+feat_phra)
    
    # 冗余剪枝
    phrases_set = []
    for i in freq_phrases_in_reviews:
        phrases_set.extend(i.split())
    phrases_set = set(phrases_set)
    # feat_dict = dict(zip(freq_nouns_in_reviews,[0]*len(freq_nouns_in_reviews)))
    feat_dict = {}
    no_superset_index = set(range(len(sentences_features))) - superset_index
    for i in no_superset_index:
        sent = sentences_features[i]
        for feat in sent:
            if feat in phrases_set:
                feat_dict.setdefault(feat,0)
                feat_dict[feat] += 1
    pruned_features = set()
    for i,j in feat_dict.items():
        if j<5:
            pruned_features.add(i)
    freq_nouns_in_reviews = list(set(freq_nouns_in_reviews)-pruned_features)
    
    # 获取高频的情感词（great,good,nice等），将含有这些词的词组删掉
    high_opinion_words = get_freq_nouns(adjs,0.08)
    drop = set()
    for i in freq_phrases_in_reviews:
        if i.split()[0] in high_opinion_words:
            drop.add(i)
    freq_phrases_in_reviews = list(set(freq_phrases_in_reviews)-drop)
    
    # 非频繁特征提取开始！！！
    infreq_feats = []
    def search_nouns(pos_sent,adj,steps):
        infreq_nouns = []
        ori_sent = [i[0] for i in pos_sent]
        idx = ori_sent.index(adj)
        rang_min = max((idx-steps),0)
        rang_max = min((idx+steps),len(ori_sent))
        for i in range(rang_min,rang_max):
            if pos_sent[i][1]=='NN' or pos_sent[i][1]=='NNS':
                infreq_nouns.append(ori_sent[i])
        return [i for i in infreq_nouns if i not in stop_words and len(i)>2]
    opinion_words = get_freq_nouns(adjs,0.005) # !!!
        # 把不含频繁feature的句子提取出来
    infreq_index = []
    for idx,sent in enumerate(all_nouns):
        temp = False
        for i in sent:
            if i in freq_nouns_in_reviews: 
                temp = True
                break
        if not temp:
            infreq_index.append(idx)
    for idx in infreq_index:
        sent = pos_results[idx]
        # 找出opinion words
        op_words = []
        for w in sent:
            if w[0] in opinion_words:
                op_words.append(w[0])
        for w in op_words:
            infreq = search_nouns(sent,w,4)
            infreq_feats.extend(infreq)
    infreq_feats = set(infreq_feats)
    # 非频繁特征提取结束！！！
    
    freq_phrases_in_reviews = set(freq_phrases_in_reviews+freq_phrases_in_reviews)|infreq_feats
    
    def real_features_set(all_feat):
        '''获取人工标注的features集合'''
        features_set = set()
        for features in all_feat:
            temp = features.split(',')
            for i in temp:
                f = re.sub('\[.+', '', i.strip())
                if f:
                    features_set.add(f)
        return features_set
    
    def finded_features_set(freq_feat):
        '''获取找到的features集合'''
        features_set = set()
        for features in freq_feat:
            if features:
                for i in features:
                    features_set.add(i)
        return features_set
    manual_features = real_features_set(real_features)
    finded_features = set(freq_nouns_in_reviews)|freq_phrases_in_reviews
    # print 'find:',finded_features
    # print '\nreal:',len(manual_features)
    print '对于%s:\n\t寻找feature的结果：' %text
    print '\t人工标记的features数量:',len(manual_features)
    print '\trecall:', len(manual_features & finded_features)/float(len(manual_features))
    print '\tprecision:', len(manual_features & finded_features)/float(len(finded_features))
    # =================================================================
    print '\t句子情感判断结果：'
    def word_synonyms_and_antonyms(word):
        '''找近、反义词'''
        synonyms=[]
        antonyms=[]
        list_good=wordnet.synsets(word)
        for syn in list_good:
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        return set(synonyms),set(antonyms)
        
    def orientation_prediction(word):
        '''判断单词的情感'''
        synonyms,antonyms = word_synonyms_and_antonyms(word)
        if word in positive_words:
            # positive_words.add(word)
            return 1
        elif word in negative_words:
            # negative_words.add(word)
            return -1
        else:
            orien = 0
            for w in antonyms:
                if w in positive_words:
                    # positive_words.add(w)
                    orien -= 1
                elif w in negative_words:
                    # negative_words.add(w)
                    orien += 1
                else:
                    orien += 0
            if orien>0: return 1
            elif orien==0: return 0
            else: return -1
    
    def orien_sent(sent):
        '''判断句子的情感'''
        orientation = 0
        for w in sent:
            orientation += orientation_prediction(w)
        if orientation>0:
            return 1
        elif orientation==0:
            return 0
        else:
            return -1
            
    def predict_orien(all_adjs,group_ori):
        '''判断整个评论中每条句子的情感'''
        predict = []
        for idx,sent in enumerate(adjs):
            ori = orien_sent(sent)
            if ori==0:
                if group_ori[get_group(idx)]>0:
                    ori = 1
                else:
                    ori = -1
            predict.append(ori)
        return predict
    
    def get_real_orientation(sent_feat):
        '''获取原始句子的情感'''
        sents_orien = []
        group_ori = dict(zip(range(len(indiv_rv_len)),[0]*len(indiv_rv_len)))
        for idx,feats in enumerate(sent_feat):
            if feats:
                result = re.findall('\[.\d]',feats)
                orien = [int(i[1:-1]) for i in result]
                if sum(orien)>0:
                    ori = 1
                else:
                    ori = -1
            else:
                ori = 0
            group_ori[get_group(idx)] += ori
            sents_orien.append(ori)
        return sents_orien,group_ori
        
    def classi_scores(y_true,y_pred):
        tp = 0
        tn = 0
        for i,j in zip(y_true,y_pred):
            if i==j==1:
                tp += 1
            if i==j==-1:
                tn += 1
        precision = float(tp)/y_pred.count(1)
        recall = float(tp)/y_true.count(1)   
        accuracy = float(tp+tn)/len(y_true)
        return precision,recall,accuracy
        
   
    sts_orientation,group_ori =  get_real_orientation(real_features) # 人工标注的情感
    prediction = predict_orien(adjs,group_ori) # 预测评论的每条句子的感情
    pure_sts_orien,pure_prediction = [],[] # 去掉没有人工标注的句子！！
    for i,j in zip(sts_orientation,prediction):
        if i != 0:
            pure_sts_orien.append(i)
            pure_prediction.append(j)
    precision,recall,accuracy = classi_scores(pure_sts_orien,pure_prediction)
    print '\tprecision: %s\n\trecall: %s\n\taccuracy: %s' %(precision,recall,accuracy)

'''<<输出结果>>：
对于Canon:
	寻找feature的结果：
	人工标记的features数量: 105
	recall: 0.438095238095
	precision: 0.4
	句子情感判断结果：
	precision: 0.866336633663
	recall: 0.940860215054
	accuracy: 0.8410041841
对于DVD:
	寻找feature的结果：
	人工标记的features数量: 116
	recall: 0.370689655172
	precision: 0.373913043478
	句子情感判断结果：
	precision: 0.675257731959
	recall: 0.873333333333
	accuracy: 0.76231884058
对于Nikon:
	寻找feature的结果：
	人工标记的features数量: 75
	recall: 0.373333333333
	precision: 0.252252252252
	句子情感判断结果：
	precision: 0.87969924812
	recall: 0.906976744186
	accuracy: 0.825
对于Nokia6610:
	寻找feature的结果：
	人工标记的features数量: 111
	recall: 0.477477477477
	precision: 0.427419354839
	句子情感判断结果：
	precision: 0.801801801802
	recall: 0.931937172775
	accuracy: 0.785714285714
对于Nomad:
	寻找feature的结果：
	人工标记的features数量: 188
	recall: 0.367021276596
	precision: 0.267441860465
	句子情感判断结果：
	precision: 0.727272727273
	recall: 0.901408450704
	accuracy: 0.742024965326
	'''
	