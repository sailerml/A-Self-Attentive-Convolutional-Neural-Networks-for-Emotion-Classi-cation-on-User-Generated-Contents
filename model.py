# -*- coding: utf-8 -*-

#TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.

# print("started...")

import tensorflow as tf

import numpy as np



class TextCNN:

    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,d_a_size, r_size, fc_size

                 ,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=True,clip_gradients=5.0,decay_rate_big=0.50):

        """init all hyperparameter here"""

        # set hyperparamter

        self.num_classes = num_classes

        self.batch_size = batch_size

        self.sequence_length=sequence_length

        self.vocab_size=vocab_size

        self.embed_size=embed_size

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate

        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, learning_rate * decay_rate_big)

        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]

        self.num_filters=num_filters

        self.initializer=initializer

        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.

        self.multi_label_flag=multi_label_flag

        self.clip_gradients = clip_gradients

        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        #for attention
        self.d_a_size = d_a_size

        self.r_size = r_size

        self.fc_size = fc_size

        # add placeholder (X,label)

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X

        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,num_classes]

        self.input_y_multilabel = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.

        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.iter = tf.placeholder(tf.int32) #training iteration

        self.tst=tf.placeholder(tf.bool)

        self.use_mulitple_layer_cnn=False



        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")

        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))

        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)

        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)

        self.decay_steps, self.decay_rate = decay_steps, decay_rate



        self.instantiate_weights()

        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.

        self.possibility=tf.nn.sigmoid(self.logits)

        if multi_label_flag:

            print("going to use multi label loss.")

            self.loss_val = self.loss_multilabel()

        else:print("going to use single label loss.");self.loss_val = self.loss()

        self.train_op = self.train()

        if not self.multi_label_flag:

            self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

            print("self.predictions:", self.predictions)

            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]

            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()



    def instantiate_weights(self):

        """define all weights here"""

        with tf.name_scope("embedding"): # embedding matrix

            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

            self.W_projection = tf.get_variable("W_projection",shape=[len(self.filter_sizes)*self.fc_size, self.num_classes],initializer=self.initializer) #[embed_size,label_size]

            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09



    def inference(self):

        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""

        # 1.=====>get emebedding of words in the sentence

        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]

        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv



        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->

        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable

        #if self.use_mulitple_layer_cnn: # this may take 50G memory.

        #    print("use multiple layer CNN")

        #    h=self.cnn_multiple_layers()

        #else: # this take small memory, less than 2G memory.

        print("use single layer CNN")

        h=self.cnn_single_layer()

        #5. logits(use linear layer)and predictions(argmax)

        with tf.name_scope("output"):

            logits = tf.matmul(h,self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])

        return logits



    def cnn_single_layer(self):

        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):

            # with tf.name_scope("convolution-pooling-%s" %filter_size):

            with tf.variable_scope("convolution-pooling-%s" % filter_size):

                # ====>a.create filter

                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)

                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.

                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`

                # Conv.Returns: A `Tensor`. Has the same type as `input`.

                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.

                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]

                # input data format:NHWC:[batch, height, width, channels];output:4-D

                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]

                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')



                # ====>c. apply nolinearity

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09

                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`


                # ====>d. self-attention
                print("h size is",h.get_shape().as_list())

                h_size = h.get_shape().as_list()

                H_reshape = tf.reshape( h, [-1,self.num_filters ]) # [batch_size,num_filters,sequence_length - filter_size + 1]

                print("new h size is", H_reshape.get_shape().as_list())

                H_forA = tf.reshape(h, [-1, h_size[1], self.num_filters])

                print("H_forA size is", H_forA.get_shape().as_list())

                self.W_s1 = tf.get_variable("W_s1", shape=[ self.num_filters, self.d_a_size], initializer=self.initializer)

                print("w1 size is", self.W_s1.get_shape().as_list())

                _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))# [batch_size,d_a_size,sequence_length - filter_size + 1]

                print("H_s1 size is", _H_s1.get_shape().as_list())

                self.W_s2 = tf.get_variable("W_s2", shape=[self.d_a_size, self.r_size], initializer=self.initializer)

                _H_s2 = tf.matmul(_H_s1, self.W_s2)#[batch_size,r_size,d_a_size]

                print("H_s2 size is", _H_s2.get_shape().as_list())

                H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, h_size[1], self.r_size]), [0, 2, 1])

                print("H_s2_reshape size is", H_s2_reshape.get_shape().as_list())

                self.A = tf.nn.softmax(H_s2_reshape, name="attention")#[batch_size,r_size,sequence_length - filter_size + 1]

                #得到卷积特征嵌入表示
                self.M = tf.matmul(self.A, H_forA)#[batch_size,r_size,num_filters]

                print("M size is", self.M.get_shape().as_list())

                #全连接
                self.M_flat = tf.reshape(self.M, shape=[-1, self.num_filters * self.r_size])#[batch_size,r_size*num_filters]

                W_fc = tf.get_variable("W_fc", shape=[self.num_filters * self.r_size, self.fc_size], initializer=self.initializer)

                b_fc = tf.Variable(tf.constant(0.1, shape=[self.fc_size]), name="b_fc")

                self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")#[batch_size,fc_size]

                h = self.fc

                #print(h.get_shape().as_list())

                pooled_outputs.append(h)
        self.h_pool = tf.concat(pooled_outputs, 1)

        print("poolout size is",self.h_pool.get_shape().as_list())
            



        # 4.=====>add dropout: use tf.nn.dropout

        with tf.name_scope("dropout"):

            self.h_drop = tf.nn.dropout(self.h_pool, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]

        h = tf.layers.dense(self.h_drop, len(self.filter_sizes)*self.fc_size, activation=tf.nn.tanh, use_bias=True)

        return h



    def cnn_multiple_layers(self):

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->

        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable

        pooled_outputs = []

        print("sentence_embeddings_expanded:",self.sentence_embeddings_expanded)

        for i, filter_size in enumerate(self.filter_sizes):

            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):

                # 1) CNN->BN->relu

                filter = tf.get_variable("filter-%s" % filter_size,[filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)

                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],padding="SAME",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]

                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')

                print(i, "conv1:", conv)

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09

                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`



                # 2) CNN->BN->relu

                h = tf.reshape(h, [-1, self.sequence_length, self.num_filters,1])  # shape:[batch_size,sequence_length,num_filters,1]

                # Layer2:CONV-RELU

                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)

                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="SAME",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]

                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')

                print(i, "conv2:", conv2)

                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09

                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`



                # 3. Max-pooling

                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1,self.sequence_length, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))

                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]

                print(i, "pooling:", pooling_max)

                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]

                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]

        # concat

        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]

        print("h.concat:", h)



        with tf.name_scope("dropout"):

            h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]

        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]



    def loss_multilabel(self,l2_lambda=0.0001): #0.0001#this loss function is for multi-label classification

        with tf.name_scope("loss"):

            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`

            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.

            #input_y:shape=(?, 1999); logits:shape=(?, 1999)

            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)

            #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)

            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).

            losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch

            loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch

            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

            loss=loss+l2_losses

        return loss



    def loss(self,l2_lambda=0.0001):#0.001

        with tf.name_scope("loss"):

            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]

            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)

            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)

            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()

            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

            loss=loss+l2_losses

        return loss



    def train(self):

        """based on the loss, use SGD to update parameter"""

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)

        self.learning_rate_=learning_rate

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01

        with tf.control_dependencies(update_ops):  #ADD 2018.06.01

            train_op = optimizer.apply_gradients(zip(gradients, variables))

        return train_op

