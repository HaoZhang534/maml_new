""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
from tqdm import trange
from glob import glob
FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot')
            record_data_folder_tr=config.get('record_data_folder_tr','./data/tfrecord/train')
            record_data_folder_te=config.get('record_data_folder_te','./data/tfrecord/test')
            self.train_files=glob(record_data_folder_tr+'/*.tfrecords')
            self.test_files=glob(record_data_folder_te+'/*.tfrecords')
            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 0
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        else:
            raise ValueError('Unrecognized data source')


    def map_fn(proto):
        features = tf.parse_single_example(
            proto, features={
                'image': tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.string)
            }
        )
        features['image'] = tf.decode_raw(features['image'], tf.float32)
        features['label'] = tf.decode_raw(features['label'], tf.int32)
        return features


    def make_data_tensor_record(self, train=True,poison=None,sess=None):
        if train:
            dataset = tf.data.TFRecordDataset(self.train_files)
            # num_total_batches = 50000
        else:
            dataset = tf.data.TFRecordDataset(self.test_files)
            # num_total_batches = 1000

        dataset=dataset.map(lambda proto:self.map_fn(proto))
        dataset = dataset.cache()
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        one_batch = iterator.get_next()
        images = tf.reshape(one_batch['image'],[self.batch_size,-1,self.dim_input])
        labels = tf.one_hot(one_batch['label'],self.num_classes)
        return images, labels


    def make_data_tensor(self, train=True,poison=None,sess=None):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 50000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 1000

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in trange(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_png(image_file)
        image.set_shape((self.img_size[0],self.img_size[1],1))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0
        image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        print('zhzhzhzhzhzhzhzhzhzhzhz')
        print(images)
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in trange(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)

            label_batch = tf.convert_to_tensor(labels)

            pred=tf.random_uniform(shape=[],minval=0,maxval=1)<FLAGS.noise_rate

            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes) #[num_classes]
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k # idx of kth sample in each class
                new_list.append(tf.gather(image_batch,true_idxs))
                # if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                #     new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                #         tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                #         k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                #         for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            # sess = tf.Session()
            if i==0 and train:
                print('hhhhhhhhhhhhhhhhhhhhhhh')
                # sess=tf.Session()
                tf.train.start_queue_runners()
                sess.run(class_idxs)
                print(new_list)
                samplex = sess.run(new_list)
                print('hhhhhhhhhhhhhhhhhhhhhhh')
                sampley = sess.run(tf.random.shuffle(new_label_list))
                print('Label flipped Label flippedLabel flippedLabel flippedLabel flippedLabel flippedLabel flippedLabel flipped')
            if ('noise' in FLAGS.mode or 'poison' in FLAGS.mode) and train:
                new_list=tf.cond(pred,lambda :poison[0],lambda :new_list)
                new_label_list=tf.cond(pred,lambda :poison[1],lambda :new_label_list)
            elif 'label_flip' in FLAGS.mode and train:
                new_list=tf.cond(pred,lambda :samplex,lambda :new_list)
                new_label_list=tf.cond(pred,lambda :sampley,lambda :new_label_list)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase