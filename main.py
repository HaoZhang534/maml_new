"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from tqdm import trange
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from math import isnan
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')
flags.DEFINE_string('poison_path', None, '')
flags.DEFINE_string('mode', 'normal_train', 'normal_train, train_poison, train_with_poison')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_float('poison_lr', 1e-2, 'step size alpha for inner gradient update.')
flags.DEFINE_float('noise_rate', 0.2, '')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('save_poison_ep', 10, 'number of inner gradient updates during training.')
flags.DEFINE_integer('poison_itr', 50, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
# flags.DEFINE_bool('stop_grad', True, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('reptile', True, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('median', True, '')
# flags.DEFINE_bool('poison', True, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_string('poison_dir', None,'')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def train(model, saver, sess, exp_string, data_generator, resume_itr=0,test_params=None):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 100

    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = 50

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in trange(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        result = sess.run(input_tensors, feed_dict)

        # pr=sess.run([model.accuraciesa])
        # print(pr)
        # print(len((pr)))
        # for it in pr:
        #     print("hhhhhhhhhhhh")
        #     print(it)
        # # print(len(pr[3][0]))
        # print(pr[2][0][0].sum())
        # # print(pr[3][0][1].shape)
        # poison=pr[1]
        # print("poison sum: %f"%(poison.sum()))
        print('Training step finished')
        if itr>=FLAGS.pretrain_iterations and FLAGS.mode=='train_poison':
            sess.run([model.poison_op])
            sess.run([model.clip_poison_op])
            print('Poison training step finished')

        if FLAGS.mode=='train_poison' and itr>0 and itr%FLAGS.save_poison_ep==0:
            np.save(FLAGS.logdir + '/' + exp_string+'/poisonx_%d.npy'%itr,sess.run(model.poisonx))
            print('Poison exampled saved')

        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL==0:
            print("Acc 1")
            print(result[-2])
            print("Acc 2")
            print(result[-1])
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
            print("Model saved")

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if ((itr!=0) and itr % TEST_PRINT_INTERVAL == 0) or itr==FLAGS.pretrain_iterations + FLAGS.metatrain_iterations-1 :
            # if 'generate' not in dir(data_generator):
            #     feed_dict = {}
            #     if model.classification:
            #         input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
            #     else:
            #         input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            # else:
            #     print('erro')
            #     exit(1)
            #
            # result = sess.run(input_tensors, feed_dict)
            # print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
            test(*test_params)

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 50

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    # out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    # out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    # with open(out_pkl, 'wb') as f:
    #     pickle.dump({'mses': metaval_accuracies}, f)
    # with open(out_filename, 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['update'+str(i) for i in range(len(means))])
    #     writer.writerow(means)
    #     writer.writerow(stds)
    #     writer.writerow(ci95)

def main():

    test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1
    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr
    exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)+ '.poison_lr' + str(FLAGS.poison_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    num_images_per_class=FLAGS.update_batch_size*3

    data_generator = DataGenerator(num_images_per_class, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    if FLAGS.mode=='train_with_poison':
        print('Loading poison examples from %s'%FLAGS.poison_path)
        poison_example=np.load(FLAGS.poison_dir)
        # poison_example=np.load(FLAGS.logdir + '/' + exp_string+'/poisonx_%d.npy'%FLAGS.poison_itr)
    else:
        poison_example=None
    model = MAML(dim_input=dim_input, dim_output=dim_output,num_images_per_class=num_images_per_class,num_classes=FLAGS.num_classes,poison_example=poison_example)
    sess = tf.InteractiveSession()
    print('Session created')
    if FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor(train=True,poison=(model.poisonx,model.poisony),sess=sess)
            if FLAGS.reptile:
                inputa=image_tensor
                labela=label_tensor
            else:
                inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
                labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
            if FLAGS.mode=='train_poison':
                inputa_test = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                inputb_test = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                labela_test = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                labelb_test = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb,'inputa_test': inputa_test, 'inputb_test': inputb_test, 'labela_test': labela_test, 'labelb_test': labelb_test}
            else:
                input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix=FLAGS.mode)

    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    print('Model built')
    model.summ_op = tf.summary.merge_all()
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)


    resume_itr = 0
    model_file = None
    tf.train.start_queue_runners()
    tf.global_variables_initializer().run()


    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    test_params=[model, saver, sess, exp_string, data_generator, test_num_updates]
    test(model, saver, sess, exp_string, data_generator, test_num_updates)
    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr,test_params=test_params)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
