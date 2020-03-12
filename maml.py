""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input, dim_output, num_images_per_class, num_classes,poison_num=1,poison_example=None):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.poison_lr=FLAGS.poison_lr
        self.classification = False
        self.poison_num=poison_num
        self.num_classes=num_classes
        self.num_images_per_task=num_images_per_class*num_classes

        if FLAGS.train:
            if FLAGS.mode=='train_with_poison':
                self.poisonx = tf.Variable(poison_example, trainable=True,dtype='float32')
            elif FLAGS.mode=='train_poison':
                self.poisonx = tf.get_variable("poisonx",
                                            shape=[self.num_images_per_task, self.dim_input],
                                            initializer=tf.zeros_initializer, trainable=True,dtype='float32')
            elif FLAGS.mode=='train_with_noise':
                self.poisonx = tf.random_uniform(name='noise',
                                                  shape=[self.num_images_per_task, self.dim_input],
                                                  minval=0.0, maxval=1.0,dtype='float32')
            else:
                self.poisonx=None
        else:
            self.poisonx=None
        self.poisony =tf.range(self.num_images_per_task,dtype='int32') % self.num_classes
        # print('jjjjjjjjjjjjjjjjjjjjjj')
        # print(self.poisony.shape)

        if FLAGS.datasource == 'omniglot':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights

            self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        else:
            raise ValueError('Unrecognized data source.')


    def construct_model(self, input_tensors, prefix='metatrain_'):
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']
        ###session for debug###########
        # self.sess=tf.Session()
        ##################################
        # if 'poison' in prefix:
        #         # self.poisonx2=tf.get_variable("poisonx2",shape=[self.poison_num,self.num_images_per_task,self.dim_input],initializer=tf.truncated_normal_initializer,trainable=True)
        #     # self.poisonx=tf.clip_by_value(self.poisonx_,0.0,1.0)
        #     # self.poisonx=tf.minimum(tf.maximum(self.poisonx,0.0),1.0)
        #     self.poisony=tf.one_hot(tf.tile(tf.reshape(tf.range(self.num_images_per_task)%self.num_classes,[1,self.num_images_per_task]),[self.poison_num,1]),depth=self.num_classes)
        #     self.inputa = tf.concat([self.poisonx1,self.inputa[self.poison_num:,:,:]],0)
        #     self.labela = tf.concat([self.poisony,self.labela[self.poison_num:,:,:]],0)
        #     self.inputb = tf.concat([self.poisonx2,self.inputb[self.poison_num:,:,:]],0)
        #     self.labelb = tf.concat([self.poisony,self.labelb[self.poison_num:,:,:]],0)
        if prefix=='train_poison':
            self.inputa_test = input_tensors['inputa_test']
            self.inputb_test = input_tensors['inputb_test']
            self.labela_test = input_tensors['labela_test']
            self.labelb_test = input_tensors['labelb_test']
        # elif prefix=='train_with_noise':
        #     self.poisonx1=tf.random_uniform(name='noise1',shape=[self.poison_num,self.num_images_per_task,self.dim_input],minval=0.0,maxval=1.0)
        #     self.poisonx2=tf.random_uniform(name='noise2',shape=[self.poison_num,self.num_images_per_task,self.dim_input],minval=0.0,maxval=1.0)
        #     self.poisony = tf.one_hot(
        #         tf.tile(tf.reshape(tf.range(self.num_images_per_task) % self.num_classes, [1, self.num_images_per_task]),
        #                 [self.poison_num, 1]), depth=self.num_classes)
        #     self.inputa = tf.concat([self.poisonx1, self.inputa[self.poison_num:, :, :]], 0)
        #     self.inputb = tf.concat([self.poisonx2, self.inputb[self.poison_num:, :, :]], 0)
        #     self.labela = tf.concat([self.poisony, self.labela[self.poison_num:, :, :]], 0)
        #     self.labelb = tf.concat([self.poisony, self.labelb[self.poison_num:, :, :]], 0)


        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()

            else:
                # Define the weights
                self.weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            # lossesa, outputas, lossesb, outputbs = [], [], [], []
            # accuraciesa, accuraciesb = [], []
            num_updates = FLAGS.num_updates
            # outputbs = [[]]*num_updates
            # lossesb = [[]]*num_updates
            # accuraciesb = [[]]*num_updates

            def task_metalearn_(inp, weights,reuse=True,stop_grad=True,fw=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                # print("inputa hhhhhhhhhhh")
                # print(inputa.shape)
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))
                # deta=dict(zip(fast_weights.keys(),[fast_weights[key]-weights[key] for key in fast_weights.keys()]))
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])
                if fw:
                    task_output.append(fast_weights)

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn_(inp=(self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]),weights=self.weights, reuse=False,fw=False)

            if prefix!='train_poison' and (not FLAGS.reptile):
                def task_metalearn(inp):
                    return task_metalearn_(inp,weights=self.weights,stop_grad=True,fw=False,reuse=True)
                out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
                if self.classification:
                    out_dtype.extend([tf.float32, [tf.float32]*num_updates])
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                if self.classification:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb =result
                else:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb =result
            else:
                if prefix == 'train_poison':
                    sg = False
                else:
                    sg = True
                if self.classification:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb,new_weights = [],[],[],[],[],[],[]
                    for itr in range(FLAGS.meta_batch_size):
                        a,b,c,d,e,f,g=task_metalearn_(inp=(self.inputa[itr], self.inputb[itr], self.labela[itr], self.labelb[itr]),weights=self.weights,reuse=True,stop_grad=sg)
                        outputas.append(a)
                        outputbs.append(b)
                        lossesa.append(c)
                        lossesb.append(d)
                        accuraciesa.append(e)
                        accuraciesb.append(f)
                        new_weights.append(g)
                    self.accuraciesa=accuraciesa
                    lossesb = tf.transpose(lossesb,[1,0,2])
                    accuraciesb = tf.transpose(accuraciesb,[1,0])
                else:
                    outputas, outputbs, lossesa, lossesb,new_weights= [], [], [], [],[]
                    for itr in range(FLAGS.meta_batch_size):
                        a, b, c, d ,e= task_metalearn_(inp=(self.inputa[itr], self.inputb[itr], self.labela[itr], self.labelb[itr]),weights=self.weights,reuse=True,stop_grad=sg)
                        outputas.append(a)
                        outputbs.append(b)
                        lossesa.append(c)
                        lossesb.append(d)
                        new_weights.append(e)
                    lossesb = tf.transpose(lossesb,[1,0,2])

            ## Performance & Optimization
            if 'train' in prefix:
                def get_median(v):
                    mid = v.get_shape()[-1] // 2 + 1
                    return tf.nn.top_k(v, mid).values[...,-1]
                if FLAGS.median:
                    new_weights = dict(zip(self.weights.keys(),
                                           [get_median(tf.concat([tf.expand_dims(weight[key],-1) for weight in new_weights],axis=-1)) for key
                                            in self.weights.keys()]))
                else:
                    new_weights=dict(zip(self.weights.keys(),[sum([ weight[key] for weight in new_weights])/FLAGS.meta_batch_size for key in self.weights.keys()]))
                new_weights=dict(zip(self.weights.keys(),[self.weights[key]+self.meta_lr*(new_weights[key]-self.weights[key]) for key in self.weights.keys()]))
                reptile_ops=[tf.assign(self.weights[key], new_weights[key]) for key in self.weights.keys() ]
                reptile_op=tf.group(*reptile_ops)
                # if not FLAGS.reptile:
                self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                # after the map_fn
                self.outputas, self.outputbs = outputas, outputbs
                if self.classification:
                    self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                    self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1,var_list=list(self.weights.values()))

                if FLAGS.metatrain_iterations > 0:
                    if FLAGS.reptile:
                        self.metatrain_op=reptile_op
                    else:
                        optimizer = tf.train.AdamOptimizer(self.meta_lr)
                        self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates - 1],
                                                                     var_list=list(self.weights.values()))
                        if FLAGS.datasource == 'miniimagenet':
                            gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                        self.metatrain_op = optimizer.apply_gradients(gvs)
                if prefix=='train_poison':
                    assert FLAGS.reptile
                    if FLAGS.reptile:
                        weights_star=new_weights
                    else:
                        gvs = tf.gradients(self.total_losses2[FLAGS.num_updates - 1],list(self.weights.values()))
                        if FLAGS.datasource == 'miniimagenet':
                            gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                        gradients = dict(zip(self.weights.keys(), gvs))
                        weights_star = dict(
                            zip(self.weights.keys(), [self.weights[key] - self.meta_lr * gradients[key] for key in self.weights.keys()]))
                    def task_metalearn_star(inp):
                        return task_metalearn_(inp, weights=weights_star, stop_grad=True,fw=False)

                    # self.metatrain_op = optimizer.apply_gradients(gvs)
                    out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
                    if self.classification:
                        out_dtype.extend([tf.float32, [tf.float32] * num_updates])
                    result = tf.map_fn(task_metalearn_star, elems=(self.inputa_test, self.inputb_test, self.labela_test, self.labelb_test),
                                       dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                    if self.classification:
                        outputas_test, outputbs_test, lossesa_test, lossesb_test, accuraciesa_test, accuraciesb_test = result
                    else:
                        outputas_test, outputbs_test, lossesa_test, lossesb_test = result
                    self.total_losses2_test = total_losses2_test = [tf.reduce_sum(lossesb_test[j]) / tf.to_float(FLAGS.meta_batch_size) for
                                                          j in range(num_updates)]
                    optimizer = tf.train.AdamOptimizer(self.poison_lr)
                    self.gvs_poison = gvs_poison = optimizer.compute_gradients(-self.total_losses2_test[FLAGS.num_updates - 1],var_list=[self.poisonx])
                    if FLAGS.datasource == 'miniimagenet':
                        gvs_poison = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs_poison]
                    self.poison_op = optimizer.apply_gradients(gvs_poison)
                    clipped_poison=tf.clip_by_value(self.poisonx,0.0,1.0)
                    self.clip_poison_op=tf.assign(self.poisonx,clipped_poison)
            else:
                self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                if self.classification:
                    self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                    self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']


