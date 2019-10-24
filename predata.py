# -*- coding: utf-8 -*-

# ********************************************************************* #

#

# ********************************************************************* #
import numpy as np
import os
import pickle
import re
import xml.dom.minidom
import jieba
import string


# ********************************************************************* #
#emotionWorsPath = './data/emotionWorsDict.json'#在含情绪文本中做情绪分类
#WorsPath = './data/WorsDict.json'#在所有文本中做情绪分类（加入不含情绪文本做噪声）
#labelPath = './data/labelDict.json'#标签
stoppath = ''
# ********************************************************************* #

PAD_ID = 0
_GO="_GO"
_END="_END"
_PAD="_PAD"
value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')       # 定义正则表达式，只取-1到+1之间的数

def toParagaphProcess(classfyDataPath):
    # 参数配置
    dom = xml.dom.minidom.parse(classfyDataPath)
    root = dom.documentElement
    dataSet = []
    blogs = root.getElementsByTagName('weibo')
    for blog in blogs:
        weiboData = {"Sub-task-ID": "", "System-ID": "", "run-tag": "", "run-type": "", "weibo-ID": "", "emotion-tag": "", "emotion_1-type": "", "emotion_2-type": "","content": []
                     }
        weiboData["Sub-task-ID"] = "1"
        weiboData["System-ID"] = "1"
        weiboData["run-tag"] = "C"
        weiboData["run-type"] = "1"
        weiboData["weibo-ID"] = blog.getAttribute("id")
        weiboData["emotion_1-type"] = blog.getAttribute("emotion-type1")
        weiboData["emotion_2-type"] = blog.getAttribute("emotion-type2")
        sentences = blog.getElementsByTagName('sentence')
        if blog.getAttribute("emotion-type1") == "none":
            weiboData['emotion-tag'] = "N"
        else:
            weiboData['emotion-tag'] = "Y"
        for sentence in sentences:
            words = list(jieba.cut(sentence.firstChild.data))
            weiboData["content"] += words#weiboData['content'] = [[word1,word2,...],[word1,word2,...]](层级网络时数据需要使用此种格式)
        dataSet.append(weiboData)
    return dataSet





#仅对含有情绪的段落文本进行分类
def create_voabulary_for_containEmotion_paragragh(datas,simple=None,name_scope=''): #zhihu-word2vec-multilabel.bin-100

    vocabulary_word2index={}
    vocabulary_index2word={}
    word_unique = {}

    vocabulary_word2index['PAD_ID']=0
    vocabulary_index2word[0]='PAD_ID'
    special_index=0
    num = 0
    if 'biLstmTextRelation' in name_scope:
        vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
        vocabulary_index2word[1]='EOS'
        special_index=1

    for i,line in enumerate(datas):
        if line['emotion-tag'] == 'Y':         
            for word in line['content']:
                # if word not in stop_words:
                if word != '\t' and word != ' ' and word_unique.get(word,None) is None:
                    num += 1
                    vocabulary_word2index[word]=num+special_index
                    vocabulary_index2word[num+special_index]=word
                    word_unique[word] = word

    # 保存字典，下次可以直接载入
    # with open(emotionWorsPath, 'wb+') as f:
    #     pickle.dump((vocabulary_word2index,vocabulary_index2word), f)
    #     f.close()
    return vocabulary_word2index,vocabulary_index2word




#对所有句子（包括不含情绪的数据文本）文本进行分类
def create_voabulary_for_all_paragragh(datas,simple=None,name_scope=''): #zhihu-word2vec-multilabel.bin-100

    vocabulary_word2index={}
    vocabulary_index2word={}
    word_unique = {}

    vocabulary_word2index['PAD_ID']=0
    vocabulary_index2word[0]='PAD_ID'
    special_index=0
    num = 0
    if 'biLstmTextRelation' in name_scope:
        vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
        vocabulary_index2word[1]='EOS'
        special_index=1

    for i,line in enumerate(datas):  
        
        for word in line['content']:
            # if word not in stop_words:
            if word != '\t' and word != ' ' and word_unique.get(word,None) is None:
                num += 1
                vocabulary_word2index[word]=num+special_index
                vocabulary_index2word[num+special_index]=word
                word_unique[word] = word

    #保存字典，下次可以直接载入
    # with open(emotionWorsPath, 'wb+') as f:
    #     pickle.dump((vocabulary_word2index,vocabulary_index2word), f)
    #     f.close()
    return vocabulary_word2index,vocabulary_index2word




# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_allparagragh_label(datas,name_scope='',use_seq2seq=False):##将不含有情绪的文本分类为"none"类

    vocabulary_word2index_label={}
    vocabulary_index2word_label={}
    vocabulary_label_count_dict={} #{label:count}
    for i,line in enumerate(datas):       
        

        label1 = line["emotion_1-type"]
        label2 = line["emotion_2-type"]
        if vocabulary_label_count_dict.get(label1,None) is not None:
            vocabulary_label_count_dict[label1] += 1
        else:
            vocabulary_label_count_dict[label1] = 0
        #标记第二个情绪
        if label2 == "none":
            continue
        elif vocabulary_label_count_dict.get(label2,None) is not None:
            vocabulary_label_count_dict[label2] += 1
        else:
            vocabulary_label_count_dict[label2] = 0
        # if line["emotion-tag"] == 'N':
        #     label3 = "none"
        #     if vocabulary_label_count_dict.get(label3,None) is not None:
        #         vocabulary_label_count_dict[label3] += 1
        #     else:
        #         vocabulary_label_count_dict[label3] = 0



    list_label=sort_by_value(vocabulary_label_count_dict)

    print("length of list_label:",len(list_label))#print(";list_label:",list_label)
    countt=0

    ##########################################################################################
    if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
        i_list=[0,1,2];label_special_list=[_GO,_END,_PAD]
        for i,label in zip(i_list,label_special_list):
            vocabulary_word2index_label[label] = i
            vocabulary_index2word_label[i] = label
    #########################################################################################
    for i,label in enumerate(list_label):
        if i<10:
            count_value=vocabulary_label_count_dict[label]
            print("label:",label,"count_value:",count_value)
            countt=countt+count_value
        indexx = i + 3 if use_seq2seq else i
        vocabulary_word2index_label[label]=indexx
        vocabulary_index2word_label[indexx]=label
    print("count top10:",countt)


    # with open(labelPath, 'wb+') as data_f:
    #     pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    # print("create_voabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label



# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_eparagragh_label(datas,name_scope='',use_seq2seq=False):#仅分类为情绪（不含none类）

    vocabulary_word2index_label={}
    vocabulary_index2word_label={}
    vocabulary_label_count_dict={} #{label:count}
    for i,line in enumerate(datas):
        if line['emotion-tag'] == 'Y':
           
            label1 = line["emotion_1-type"]
            label2 = line["emotion_2-type"]
            if vocabulary_label_count_dict.get(label1,None) is not None:
                vocabulary_label_count_dict[label1] += 1
            else:
                vocabulary_label_count_dict[label1] = 0
            #标记第二个情绪
            if label2 == "none":
                continue
            elif vocabulary_label_count_dict.get(label2,None) is not None:
                vocabulary_label_count_dict[label2] += 1
            else:
                vocabulary_label_count_dict[label2] = 0

    list_label=sort_by_value(vocabulary_label_count_dict)

    print("length of list_label:",len(list_label))#print(";list_label:",list_label)
    countt=0

    ##########################################################################################
    if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
        i_list=[0,1,2];label_special_list=[_GO,_END,_PAD]
        for i,label in zip(i_list,label_special_list):
            vocabulary_word2index_label[label] = i
            vocabulary_index2word_label[i] = label
    #########################################################################################
    for i,label in enumerate(list_label):
        if i<10:
            count_value=vocabulary_label_count_dict[label]
            print("label:",label,"count_value:",count_value)
            countt=countt+count_value
        indexx = i + 3 if use_seq2seq else i
        vocabulary_word2index_label[label]=indexx
        vocabulary_index2word_label[indexx]=label
    print("count top10:",countt)


    # with open(labelPath, 'wb+') as data_f:
    #     pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    # print("create_voabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label


#将LABEL转化为MULTI-HOT
def transform_multilabel_as_multihot(label_list,label_size=8): #1999label_list=[0,1,4,9,5]
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]



def load_data_multilabel_eparagragh_new(vocabulary_word2index,vocabulary_word2index_label,datas,max_training_data=1000000,valid_portion=0.3,multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6):  # 仅在含有情绪文本中分类。此时datas包含测试集与训练集，valid_portion给出训练集比例
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
   
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input=[] #ADD 2017-06-15
    num = -1 #数据集计数
    for i, line in enumerate(datas):

        if line['emotion-tag'] == 'Y':

            num += 1
            x = list()
            y = list()
            for word in line['content']:
                    # if word not in stop_words:
                if word != '\t' and word != ' ':
                    x.append(word)
            y.append(line["emotion_1-type"])
            if line["emotion_2-type"] != "none":
                y.append(line["emotion_2-type"])
            # print("x",x)
            # print("y",y)

            # x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
            # y=y.strip().replace('\n','')
            #x = x.strip()
            if num<1:
                print(num,"x0:",x) #get raw x
            #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
            #x=x.split(" ")
            x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
            if num<2:
                print(num,"x1:",x) #word to index
            if use_seq2seq:        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
                #ys = y.replace('\n', '').split(" ")  # ys is a list
                ys = y
                _PAD_INDEX=vocabulary_word2index_label[_PAD]
                ys_mulithot_list=[_PAD_INDEX]*seq2seq_label_length #[3,2,11,14,1]
                ys_decoder_input=[_PAD_INDEX]*seq2seq_label_length
                # below is label.
                for j,y in enumerate(ys):
                    if j<seq2seq_label_length-1:
                        ys_mulithot_list[j]=vocabulary_word2index_label[y]
                if len(ys)>seq2seq_label_length-1:
                    ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]#ADD END TOKEN
                else:
                    ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

                # below is input for decoder.
                ys_decoder_input[0]=vocabulary_word2index_label[_GO]
                for j,y in enumerate(ys):
                    if j < seq2seq_label_length - 1:
                        ys_decoder_input[j+1]=vocabulary_word2index_label[y]
                if num<10:
                    print(num,"ys:==========>0", ys)
                    print(num,"ys_mulithot_list:==============>1", ys_mulithot_list)
                    print(num,"ys_decoder_input:==============>2", ys_decoder_input)
            else:
                if multi_label_flag: # 2)prepare multi-label format for classification
                    #ys = y.replace('\n', '').split(" ")  # ys is a list
                    ys = y
                    #print(ys)
                    ys_index=[]
                    for y in ys:
                        y_index = vocabulary_word2index_label[y]
                        ys_index.append(y_index)
                    ys_mulithot_list=transform_multilabel_as_multihot(ys_index)
                else:                #3)prepare single label format for classification
                    ys_mulithot_list=vocabulary_word2index_label[y]
            if num<=3:
                print("ys_index:")
                #print(ys_index)
                print(i,"y:",y," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
            X.append(x)
            Y.append(ys_mulithot_list)
            if use_seq2seq:
                Y_decoder_input.append(ys_decoder_input) #decoder input
                #if i>50000:
                #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:",number_examples) #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    # trainX = X[0:int((1 - valid_portion) * number_examples)]
    # trainY =  Y[0:int((1 - valid_portion) * number_examples)]
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # testX = X[int((1 - valid_portion) * number_examples) + 1:]
    # testY = Y[int((1 - valid_portion) * number_examples) + 1:]
    if use_seq2seq:
        train=train+(Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test=test+(Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("load_data.ended...")
    return train,test,test



def load_data_multilabel_allparagragh_new(vocabulary_word2index,vocabulary_word2index_label,datas,max_training_data=1000000,valid_portion=0.3,multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6):  # 仅在含有情绪文本中分类。此时datas包含测试集与训练集，valid_portion给出训练集比例
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
   
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    seq_count = {}
    Y_decoder_input=[] #ADD 2017-06-15
    num = -1 #数据集计数
    for i, line in enumerate(datas):
            
        num += 1
        x = list()
        y = list()
        for word in line['content']:
                # if word not in stop_words:
            if word != '\t' and word != ' ':
                x.append(word)

        seqLength = len(x)
        if seq_count.get(seqLength, None) is not None:
            seq_count[seqLength] += 1
        else:
            seq_count[seqLength] = 0

        y.append(line["emotion_1-type"])
        if line["emotion_2-type"] != "none":
            y.append(line["emotion_2-type"])



        # x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        # y=y.strip().replace('\n','')
        #x = x.strip()
        if num<1:
            print(num,"x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        #x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if num<2:
            print(num,"x1:",x) #word to index
        if use_seq2seq:        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
            #ys = y.replace('\n', '').split(" ")  # ys is a list
            ys = y
            _PAD_INDEX=vocabulary_word2index_label[_PAD]
            ys_mulithot_list=[_PAD_INDEX]*seq2seq_label_length #[3,2,11,14,1]
            ys_decoder_input=[_PAD_INDEX]*seq2seq_label_length
            # below is label.
            for j,y in enumerate(ys):
                if j<seq2seq_label_length-1:
                    ys_mulithot_list[j]=vocabulary_word2index_label[y]
            if len(ys)>seq2seq_label_length-1:
                ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]#ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0]=vocabulary_word2index_label[_GO]
            for j,y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j+1]=vocabulary_word2index_label[y]
            if num<10:
                print(num,"ys:==========>0", ys)
                print(num,"ys_mulithot_list:==============>1", ys_mulithot_list)
                print(num,"ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag: # 2)prepare multi-label format for classification
                #ys = y.replace('\n', '').split(" ")  # ys is a list
                ys = y
                #print(ys)
                ys_index=[]
                for y in ys:
                    y_index = vocabulary_word2index_label[y]
                    ys_index.append(y_index)
                ys_mulithot_list=transform_multilabel_as_multihot(ys_index)
            else:                #3)prepare single label format for classification
                ys_mulithot_list=vocabulary_word2index_label[y]
        if num<=3:
            print("ys_index:")
            #print(ys_index)
            print(i,"y:",y," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input) #decoder input
        #if i>50000:
        #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:",number_examples) #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    # trainX = X[0:int((1 - valid_portion) * number_examples)]
    # trainY =  Y[0:int((1 - valid_portion) * number_examples)]
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # testX = X[int((1 - valid_portion) * number_examples) + 1:]
    # testY = Y[int((1 - valid_portion) * number_examples) + 1:]
    if use_seq2seq:
        train=train+(Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test=test+(Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("----------seq count is",seq_count)
    print("load_data.ended...")
    return train,test,test



