# -*- coding: utf-8 -*-



import tensorflow as tf

import numpy as np

from model import TextCNN

from tflearn.data_utils import to_categorical, pad_sequences
from predata import toParagaphProcess,create_voabulary_allparagragh_label,create_voabulary_for_all_paragragh,load_data_multilabel_allparagragh_new


import pickle

from sklearn.metrics import label_ranking_average_precision_score,recall_score, f1_score,precision_score,precision_recall_fscore_support
import os

import random

import re

#configuration

FLAGS=tf.app.flags.FLAGS


#add your own data path
# ********************************************************************* #
trainDataPath = "D:/code/NLPdataset/weiboemotion/Training data for Emotion Classification.xml"#150000
testDataPath = "D:/code/NLPdataset/weiboemotion/EmotionClassficationTest.xml"#150000
word2vectorPath = "D:/code/NLPdataset/wordvector/sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5"

# ********************************************************************* #

revalue = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')       # 定义正则表达式，只取-1到+1之间的数

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")

tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #批处理的大小 32-->128

tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128

tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少

tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")

tf.app.flags.DEFINE_integer("sentence_len",150,"max sentence length")

tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")

##for self-attention
tf.app.flags.DEFINE_integer("d_a_size", 350, "Size of W_s1 embedding")

tf.app.flags.DEFINE_integer("r_size", 30, "Size of W_s2 embedding")

tf.app.flags.DEFINE_integer("fc_size", 2000, "Size of fully connected layer")
############

tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")

tf.app.flags.DEFINE_integer("num_epochs",200,"number of epochs to run.")

tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证


tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")

tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512

tf.app.flags.DEFINE_string("training_path",trainDataPath, "location of traning data.")

tf.app.flags.DEFINE_string("text_path",testDataPath, "location of traning data.")

tf.app.flags.DEFINE_string("word2vec_model_path", word2vectorPath, "word2vec's vocabulary and vectors")

tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")

tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")

filter_sizes=[4,7,9]
#filter_sizes=[12]


#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

def main(_):

    #trainX, trainY, testX, testY = None, None, None, None

    #vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope)
    traindatas = toParagaphProcess(FLAGS.training_path)

    testdatas = toParagaphProcess(FLAGS.text_path)


    totaldatas = traindatas + testdatas

   
    trainX, trainY, testX, testY = None, None, None, None

    #word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)

    vocabulary_word2index, vocabulary_index2word = create_voabulary_for_all_paragragh(totaldatas)

    vocab_size = len(vocabulary_word2index)

    print("cnn_model.vocab_size:",vocab_size)

    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_allparagragh_label(totaldatas)

    num_classes=len(vocabulary_word2index_label)

    print("num_classes:",num_classes)

    train, test, _ = load_data_multilabel_allparagragh_new(vocabulary_word2index, vocabulary_word2index_label, totaldatas,multi_label_flag=FLAGS.multi_label_flag)

    
    trainX, trainY = train #TODO trainY1999

    testX, testY = test #TODO testY1999


    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length

    print("end padding & transform to one hot...")
    num_examples, FLAGS.sentence_len=trainX.shape

    print("trainX[0:10]:", trainX[0:10])

    print("trainY[0]:", trainY[0:10])

    train_y_short = get_target_label_short(trainY[0])

    print("train_y_short:", train_y_short)



    #2.create session.

    config=tf.ConfigProto()

    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        #Instantiate Model

        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,

                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.d_a_size,FLAGS.r_size,FLAGS.fc_size,multi_label_flag=FLAGS.multi_label_flag)

        #Initialize Save

        saver=tf.train.Saver()

        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):

            print("Restoring Variables from Checkpoint.")

            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            #for i in range(3): #decay learning rate if necessary.

            #    print(i,"Going to decay learning rate by half.")

            #    sess.run(textCNN.learning_rate_decay_half_op)

        else:

            print('Initializing Variables')

            sess.run(tf.global_variables_initializer())

            if FLAGS.use_embedding: #load pre-trained word embedding

                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, FLAGS.word2vec_model_path)

        curr_epoch=sess.run(textCNN.epoch_step)

        #3.feed data & training

        number_of_training_data=len(trainX)

        batch_size=FLAGS.batch_size

        iteration=0

        for epoch in range(curr_epoch,FLAGS.num_epochs):

            loss, counter =  0.0, 0

            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):

                iteration=iteration+1

                if epoch==0 and counter==0:

                    print("trainX[start:end]:",trainX[start:end])

                feed_dict = {textCNN.input_x: trainX[start:end],textCNN.dropout_keep_prob: 0.8,textCNN.is_training_flag:FLAGS.is_training_flag}

                if not FLAGS.multi_label_flag:

                    feed_dict[textCNN.input_y] = trainY[start:end]

                else:

                    feed_dict[textCNN.input_y_multilabel]=trainY[start:end]

                curr_loss,lr,predictions,_=sess.run([textCNN.loss_val,textCNN.learning_rate,textCNN.logits,textCNN.train_op],feed_dict)

                loss,counter=loss+curr_loss,counter+1

                if counter %100==0:
                    samples = np.array(trainY[start:end])
                    finalArr = label_ranking_average_precision_score(samples, predictions)
                    
                    binaryLabel = []
                    for i, sample in enumerate(samples):
                        arrt = filter(lambda x: x != 0, sample)
                        topnum = len(list(arrt))

                        binaryLabel.append(getRealLab(predictions[i], top_number=topnum))

                    binaryLabel = np.array(binaryLabel).astype(int)
                    microF = []
                    macroF = []
                    for sy_true, sy_pred in zip(samples, binaryLabel):
                        F1 = f1_score(sy_true, sy_pred, average='micro')
                        F2 = f1_score(sy_true, sy_pred, average='macro')
                        microF.append(F1)
                        macroF.append(F2)
                    arrMi = np.array(microF)
                    arrMa = np.array(macroF)
                    finalMiF = np.mean(arrMi)
                    finalMaF = np.mean(arrMa)
                    print(
                        "Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain rank Acc:%.3f\tmicroF score:%.3f\tmacroF score:%.3f" % (
                            epoch, counter, loss / float(counter), finalArr, finalMiF, finalMaF))


                ########################################################################################################

                if start%(3000*FLAGS.batch_size)==0: # eval every 3000 steps.

                    # eval_loss, f1_score,f1_micro,f1_macro = do_eval(sess, textCNN, testX, testY, num_classes)

                    # print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch, eval_loss, f1_score,f1_micro,f1_macro))
                    eval_loss, eval_acc, microF, macroF = do_eval(sess, textCNN, testX, testY, batch_size,
                                                                              vocabulary_index2word_label)
                    # visualize(sess, textCNN, testX, testY, random.randint(0, batch_size), vocabulary_index2word_label, vocabulary_index2word)

                    # print(
                    #     "textCNN==>Epoch %d Validation Loss:%.3f\tValidation rank Accuracy: %.3f\tValidation microF score: %.3f\tValidation macroF score: %.3f" % (
                    #         epoch, eval_loss, eval_acc, microF, macroF))

                    save_path = FLAGS.ckpt_dir + "model.ckpt"

                    print("Going to save model..")

                    saver.save(sess, save_path, global_step=epoch)

                ########################################################################################################

            #epoch increment

            print("going to increment epoch counter....")

            sess.run(textCNN.epoch_increment)



            # 4.validation

            # print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            #
            # if epoch % FLAGS.validate_every==0:
            #     eval_loss, eval_acc, microF, macroF = do_eval(sess, textCNN, testX, testY, batch_size,
            #                                                               vocabulary_index2word_label)
            #     # visualize(sess, textCNN, testX, testY, random.randint(0, batch_size), vocabulary_index2word_label, vocabulary_index2word)
            #
            #     print(
            #         "textCNN==>Epoch %d Validation Loss:%.3f\tValidation rank Accuracy: %.3f\tValidation microF score: %.3f\tValidation macroF score: %.3f" % (
            #             epoch, eval_loss, eval_acc, microF, macroF))
            #
            #     save_path=FLAGS.ckpt_dir+"model.ckpt"
            #
            #     saver.save(sess,save_path,global_step=epoch)



        # 5.最后在测试集上做测试，并报告测试准确率 Test

        eval_loss, eval_acc, microF, macroF = do_eval(sess, textCNN, testX, testY, batch_size,
                                                          vocabulary_index2word_label)
        print(
            "textCNN==>Epoch %d test Loss:%.3f\ttest rank Accuracy: %.3f\ttest microF score: %.3f\ttest macroF score: %.3f" % (
                epoch, eval_loss, eval_acc, microF, macroF))

    pass



# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,batch_size,vocabulary_index2word_label):
    number_examples=len(evalX)
    print("the example is:",number_examples)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1,textCNN.is_training_flag:FLAGS.is_training_flag}
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel] = evalY[start:end]
        curr_eval_loss, logits= sess.run([textCNN.loss_val,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy

        samples = np.array(evalY[start:end])
        finalArr = label_ranking_average_precision_score(samples, logits)
        binaryLabel = []
        for i, sample in enumerate(samples):
            arrt = filter(lambda x: x != 0, sample)
            topnum = len(list(arrt))

            binaryLabel.append(getRealLab(logits[i], top_number=topnum))

        binaryLabel = np.array(binaryLabel).astype(int)
        microF = []
        macroF = []
        for sy_true, sy_pred in zip(samples, binaryLabel):
            F1 = f1_score(sy_true, sy_pred, average='micro')
            F2 = f1_score(sy_true, sy_pred, average='macro')
            microF.append(F1)
            macroF.append(F2)
        arrMi = np.array(microF)
        arrMa = np.array(macroF)
        finalMiF = np.mean(arrMi)
        finalMaF = np.mean(arrMa)

        eval_loss, eval_counter = eval_loss + curr_eval_loss, eval_counter + 1
    return eval_loss / float(eval_counter), finalArr, finalMiF, finalMaF

#######################################



def getRealLab(logits,top_number):
    indexList = np.argsort(logits)
    labelSize = len(logits)
    binaryLabel = np.zeros(labelSize)
    if top_number == 0:
        return binaryLabel
    index = indexList[-top_number:]
    binaryLabel[index] = 1
    return binaryLabel









def get_target_label_short(eval_y):

    eval_y_short=[] #will be like:[22,642,1391]

    for index,label in enumerate(eval_y):

        if label>0:

            eval_y_short.append(index)

    return eval_y_short







##################################################


def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,vectorPath):
    #载入词向量
    datas = open(vectorPath, encoding='gb18030', errors='ignore').readlines()
    word2vec_dict = {}
    embed_dim = 300


    for i,line in enumerate(datas):

        if i == 0:
            wordnum = line.split(" ")[0]
            print('loaded %s words' % wordnum)
            dim = int(line.split(" ")[1])
            assert dim == embed_dim,'预训练词向量维度与超参数不符！'

        elif i>0:
            word = line.split(" ")[0]
            vectors = line.split(" ")[1:]
            fVectors = []
            for num in vectors:
                if revalue.match(num):
                    temp = float(num)
                    fVectors.append(temp)

            word2vec_dict[word] = fVectors


    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
        # print(len(word_embedding_2dlist[i]))
        # if len(word_embedding_2dlist[i]) != 300:
        #     print("------------wrong-------------")

    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")




if __name__ == "__main__":

    tf.app.run()