import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.python.framework import dtypes
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from convfeatures import *
# import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import pickle
import sys
import os


class Caption_Generator():

    def __init__(self, config, data=None):
        # self.dim_imgft = np.int(config.dim_imgft)
        self.dim_imgft = 2048#1536
        self.embedding_size = np.int(config.embedding_size)
        self.num_hidden = np.int(config.num_hidden)
        self.batch_size = np.int(config.batch_size)
        self.num_timesteps = np.int(config.num_timesteps)
        self.max_len = config.max_len
        self.word_threshold = config.word_threshold
        self.bias_init = config.bias_init
        self.xavier_init = config.xavier_init
        self.dropout = config.dropout
        self.lstm_keep_prob = config.lstm_keep_prob
        self.beta = config.beta_l2
        self.mode = config.mode
        self.batch_decode = config.batch_decode
        self.learning_rate = config.learning_rate
        self.resume = config.resume
        self.savedecoder = config.savedecoder
        self.saveencoder = config.saveencoder

        if self.mode == 'train':
            self.vocab, self.wtoidx, self.features, self.captions = data
            self.num_batch = int(int(self.features.shape[0]) / self.batch_size)
            print("Batch number")
            print(self.num_batch)

            print("Converting Captions to IDs")
            self.captions = self.Words_to_IDs(self.wtoidx, self.captions)
            if self.resume:
                self.vocab = np.load("Dataset/vocab.npy", allow_pickle=True).tolist()
                self.wtoidx = np.load("Dataset/wordmap.npy",  allow_pickle=True).tolist()
            print("vocab and wordmap are loaded")

        self.current_epoch = 0
        self.current_step = 0
        if self.resume or self.mode == 'test':
            if os.path.isfile('model/save.npy'):
                self.current_epoch, self.current_step = np.load(
                    "model/save.npy")
            else:
                print("No Checkpoints, Restarting Training..")
                self.resume = False
        self.nb_epochs = 2

        if self.mode == 'test':
            self.vocab = np.load("Dataset/vocab.npy",allow_pickle=True).tolist()
            self.wtoidx = np.load("Dataset/wordmap.npy",allow_pickle=True).tolist()
            self.max_words = np.int(len(self.wtoidx))
            self.idxtow = dict(zip(self.wtoidx.values(), self.wtoidx.keys()))
            self.model()
            self.image_features, self.IDs = self.build_decode_graph()
            self.load_image=config.load_image
            if not self.batch_decode: #save encoder decoder burayı calıstırıyor
                self.io = build_prepro_graph(config.inception_path)
                self.sess = self.init_decode()
            return

        #buraya eğitimde giriyor
        self.max_words = np.int(len(self.wtoidx))
        self.idxtow = dict(zip(self.wtoidx.values(), self.wtoidx.keys()))
        self.model()


    def Words_to_IDs(self, wtoidx, caption_batch):
        for i, caption in enumerate(caption_batch):
            cap = []
            for word in caption.split():
                try:
                    cap.append(wtoidx[word])
                except KeyError:
                    cap.append(wtoidx["<UNK>"])
            caption_batch[i] = np.array(cap)

        return np.vstack(caption_batch)

    def IDs_to_Words(self, idxtow, ID_batch):
        return [idxtow[word] for IDs in ID_batch for word in IDs]

    def generate_mask(self, ID_batch, wtoidx):
        nonpadded = map(lambda x: len(
            ID_batch[0]) - x.count(wtoidx["<PAD>"]), ID_batch.tolist())
        ID_batch = np.zeros((ID_batch.shape[0], self.max_len + 2))
        nonpadded_list = list(nonpadded)
        for ind, row in enumerate(ID_batch):
            row[:nonpadded_list[ind]] = 1
        return ID_batch

    def get_next_batch(self):
        batch_size = self.batch_size
        for batch_idx in range(0, len(self.features), batch_size):
            images_batch = self.features[batch_idx:batch_idx + batch_size]
            caption_batch = self.captions[batch_idx:batch_idx + batch_size]
            # print caption_batch
            yield images_batch, caption_batch

    # From NeuralTalk by Andrej Karpathy
    def init_bias(self):
        bias_init_vector = np.array(
            [1.0 * self.vocab[self.idxtow[i]] for i in self.idxtow])
        bias_init_vector /= np.sum(bias_init_vector)
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector)
        return bias_init_vector

    def assign_weights(self, dim1, dim2=None, name=None, Xavier=False):
        if Xavier:
            weight_initializer = tf.contrib.layers.xavier_initializer()
            return tf.get_variable(
                name, [dim1, dim2], initializer=weight_initializer)
        return tf.Variable(tf.random.truncated_normal([dim1, dim2]),
                           name=name)

    def assign_biases(self, dim, name, bias_init=False):
        if bias_init:
            return tf.Variable(self.init_bias().astype(np.float32), name=name)
        return tf.Variable(tf.zeros([dim]), name=name)

    def model(self):
        self.word_embedding = {
            "weights": self.assign_weights(
                self.max_words,
                self.embedding_size,
                'Weight_emb'),
            "biases": self.assign_biases(
                self.embedding_size,
                "Bias_emb")}

        self.image_embedding = {
            "weights": self.assign_weights(
                self.dim_imgft,
                self.embedding_size,
                'Weight_img_emb'),
            "biases": self.assign_biases(
                self.embedding_size,
                'Bias_img_emb')}

        self.target_word = {
            "weights": self.assign_weights(
                self.embedding_size,
                self.max_words,
                'Weight_target'),
            "biases": self.assign_biases(
                self.max_words,
                'Bias_target', bias_init=self.bias_init)}

        # self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)
        # try with --------->> tf.compat.v1.nn.rnn_cell.LSTMCell
        # self.lstm_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_hidden)
        #
        # if self.dropout:
        #     self.lstm_cell = tf.contrib.rnn.DropoutWrapper(
        #         self.lstm_cell, self.lstm_keep_prob, self.lstm_keep_prob)

        #num_layers = 3
        #self.lstm_cells = []
        #for _ in range(num_layers):
        #    self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)
        #    if self.dropout:
        #        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(
        #            self.lstm_cell, self.lstm_keep_prob, self.lstm_keep_prob)
        #    self.lstm_cells.append(self.lstm_cell)
        #self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cells,state_is_tuple=False)

        self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)

        self.inp_dict = {
            "features": tf.placeholder(
                tf. float32, [self.batch_size, self.dim_imgft], name="Train_Features"),
            "captions": tf.placeholder(
                tf.int32, [self.batch_size, self.num_timesteps], name="Train_Captions"),
            "mask": tf.placeholder(
                tf. float32, [self.batch_size, self.num_timesteps], name="Train_Mask")
        }

    def create_feed_dict(self, Ids, features, mask, mode="train"):
        feed_dict = {}
        feed_dict[self.inp_dict['captions']] = Ids
        feed_dict[self.inp_dict['features']] = features
        feed_dict[self.inp_dict['mask']] = mask
        return feed_dict

    def build_train_graph(self):
        self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)

        init_c = tf.zeros([self.batch_size, self.lstm_cell.state_size])
        initial_state = init_c
        image_emb = tf.contrib.keras.backend.dot(self.inp_dict["features"], self.image_embedding[
            'weights']) + self.image_embedding['biases']

        with tf.variable_scope("RNN"):
            output, state = self.lstm_cell(image_emb, initial_state)
            loss = 0.0
            for i in range(1, self.num_timesteps):
                batch_embed = tf.nn.embedding_lookup(
                    self.word_embedding['weights'], self.inp_dict['captions'][
                        :, i - 1]) + self.word_embedding['biases']
                tf.get_variable_scope().reuse_variables()
                output, state = self.lstm_cell(batch_embed, state)
                words = tf.reshape(self.inp_dict['captions'][
                                   :, i], shape=[self.batch_size, 1])
                onehot_encoded = tf.one_hot(indices=words, depth=len(
                    self.wtoidx), on_value=1, off_value=0, axis=-1)
                onehot_encoded = tf.reshape(onehot_encoded, shape=[
                                            self.batch_size, self.max_words])
                target_logit = tf.contrib.keras.backend.dot(
                    output, self.target_word['weights']) + self.target_word['biases']
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=target_logit, labels=onehot_encoded)

                cross_entropy = cross_entropy * self.inp_dict["mask"][:, i]
                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss
        loss = loss / tf.reduce_sum(self.inp_dict["mask"][:, 1:])

        # introducing L2 regularization in Loss/Cost Function
        # self.beta=0
        #l2_loss = self.beta * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not "Bias" in tf_var.name])
        #loss = tf.reduce_mean(loss+l2_loss)
        return loss, self.inp_dict

    def build_decode_graph(self):

        image_features = tf.placeholder(
            tf. float32, [1, self.dim_imgft], name='Input_Features')
        image_emb = tf.contrib.keras.backend.dot(image_features, self.image_embedding[
                              'weights']) + self.image_embedding['biases']
        print("image_emb:")
        print(image_emb)
        initial_state = tf.zeros([1, self.lstm_cell.state_size])
        print("initial state:")
        print(initial_state)
        IDs = []
        with tf.variable_scope("RNN"):
            output, state = self.lstm_cell(image_emb, initial_state)
            pred_ID = tf.nn.embedding_lookup(
                self.word_embedding['weights'], [
                    self.wtoidx["<S>"]]) + self.word_embedding['biases']
            for i in range(self.num_timesteps):
                tf.get_variable_scope().reuse_variables()
                output, state = self.lstm_cell(pred_ID, state)
                logits = tf.contrib.keras.backend.dot(output, self.target_word[
                                   "weights"]) + self.target_word["biases"]
                predicted_next_idx = tf.argmax(logits, axis=1)

                pred_ID = tf.nn.embedding_lookup(
                    self.word_embedding['weights'], predicted_next_idx)
                pred_ID = pred_ID + self.word_embedding['biases']
                print("pred_ID:")
                print(pred_ID)
                predicted_next_idx = tf.cast(predicted_next_idx, tf.int32, name="word_"+str(i))
                IDs.append(predicted_next_idx)
                print("IDs:")
                print(IDs)

        with open("model/Decoder/DecoderOutputs.txt", 'w') as f:
            for name in IDs:
                f.write(name.name.split(":0")[0] + "\n")

        return image_features, IDs

    def total_params(self):
        total_parameters = 0
        for variable in tf.all_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print('Total Parameters=%s', str(total_parameters))

    def train(self, loss, inp_dict):
        self.loss = loss
        self.inp_dict = inp_dict

        saver = tf.train.Saver(max_to_keep=10)
        global_step = tf.Variable(
            self.current_step,
            name='global_step')
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate, global_step, 100000, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss, global_step=global_step)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", learning_rate)
        summary_op = tf.summary.merge_all()

        self.total_params()

        with tf.Session() as sess:
            print("Initializing Training")
            init = tf.global_variables_initializer()
            sess.run(init)
            if self.resume:
                print("nb_epochs")
                print(self.nb_epochs)
                print("Loading Previously Trained Model")
                print(self.current_epoch, "Out of", self.nb_epochs, "Completed in previous run.")
                try:
                    ckpt_file = "./model/model.ckpt-" + str(self.current_step)
                    saver.restore(sess, ckpt_file)
                    print("Resuming Training")
                except Exception as e:
                    print(str(e).split('\n')[0])
                    print("Checkpoints not found")
                    sys.exit(0)
            writer = tf.summary.FileWriter(
                "model/log_dir/", graph=tf.get_default_graph())

            for epoch in range(self.current_epoch, self.nb_epochs):
                loss=[]
                idx = np.random.permutation(self.features.shape[0])
                self.captions = self.captions[idx]
                self.features = self.features[idx]
                batch_iter = self.get_next_batch()
                for batch_idx in range(self.num_batch):
                    batch_features, batch_Ids = batch_iter.__next__()
                    batch_mask = self.generate_mask(batch_Ids, self.wtoidx)
                    run = [global_step, optimizer, self.loss, summary_op]
                    feed_dict = self.create_feed_dict(
                        batch_Ids, batch_features, batch_mask)
                    step, _, current_loss, summary = sess.run(
                        run, feed_dict=feed_dict)
                    writer.add_summary(summary, step)
                    if step % 100 == 0:
                        print(epoch, ": Global Step:", step, "\tLoss: ", current_loss)
                    loss.append(current_loss)

                print("\nEpoch: ", epoch, "\tAverage Loss: ", np.mean(loss))
                print("\nSaving Model..\n")
                saver.save(sess, "./model/model.ckpt", global_step=step)
                np.save("model/save", (epoch, step))
            print("15 epochs finished")

    def init_decode(self):
        #save encoder decoder burayı calıstırıyor
        saver = tf.compat.v1.train.Saver()
        ckpt_file = "./model/model.ckpt-" + str(self.current_step) #str(89994)
        # sess = tf.Session()
        sess = tf.compat.v1.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, ckpt_file)
        return sess

    def decode(self, path):

        # SAVE ENCODER AND DECODER tamamını calıstırıyor---------------------
        features = get_features(self.sess, self.io, path, self.saveencoder)
        caption_IDs = self.sess.run(
            self.IDs, feed_dict={
                self.image_features: features})
        sentence = " ".join(self.IDs_to_Words(self.idxtow, caption_IDs))
        sentence = sentence.split("</S>")[0]
        if self.load_image:
            plt.imshow(Image.open(path))
            plt.axis("off")
            plt.title(sentence, fontsize='10', loc='left')
            name = path.split("/")[-1]
            plt.savefig("./results/" + "gen_" + name)
            plt.show()
        else:
            print("sentence:")
            print(sentence)

        if self.savedecoder:
            saver = tf.train.Saver()
            saver.save(self.sess, "model/Decoder/model.ckpt")

        # SAVE ENCODER AND DECODER ---------------------

        # CREATE CAPTION for MSCOCO ---------------------
        # import json
        # with open("Dataset/captions_val2017.json", 'r') as f:
        #     data = json.load(f)
        #
        # filenames = sorted([d["file_name"].split('.')[0] for d in data['images']])
        # filenames = np.unique(filenames)
        # files = sorted(os.listdir(path))

        # CREATE CAPTION for FLICKR ---------------------

        #with open("Dataset/Flickr8k.token.txt", 'r') as f:
        #    data = f.readlines()
        #filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
        #filenames = np.unique(filenames)
        #files = sorted(os.listdir(path))

        #files = files[4972:]
        #image_id = 4972
        # for im in enumerate(files):
        #     image = 'Dataset/flickr8k-images/' + im[1]    #image = 'Dataset/val2017/' + im[1]
        #     print(image)
        #     features = get_features(self.sess, self.io, image, self.saveencoder)
        #     with open("model/Decoder/Generated_Captions_Flickr8k.txt", 'a') as caption_file:
        #         for i, feat in enumerate(features):
        #             feat = np.reshape(feat, newshape=(1, OUTPUT_SIZE))
        #             caption_IDs = self.sess.run(
        #                 self.IDs, feed_dict={
        #                     self.image_features: feat})
        #             sentence = " ".join(
        #                 self.IDs_to_Words(
        #                     self.idxtow, caption_IDs))
        #             sentence = sentence.split("</S>")[0]
        #             print(sentence)
        #             caption_file.write(filenames[image_id] + "\t" + sentence + "\n")
        #             print("image_id:{}".format(image_id))
        #         image_id += 1
        # print("done")
        # CREATE CAPTION ---------------------


        # import json
        # with open("Dataset/captions_val2017.json", 'r') as f:
        #     data = json.load(f)
        #
        # filenames = sorted([d["file_name"].split('.')[0] for d in data['images']])
        # filenames = np.unique(filenames)
        # files = sorted(os.listdir(path))
        #
        # for im in enumerate(files):
        #     image = 'Dataset/val2017/' + im[1]
        #     if image == 'Dataset/val2017/000000561223.jpg':
        #         print(image)
        #         features = get_features(self.sess, self.io, image, self.saveencoder)
        #
        #         for i, feat in enumerate(features):
        #             feat = np.reshape(feat, newshape=(1, OUTPUT_SIZE))
        #             caption_IDs = self.sess.run(
        #                 self.IDs, feed_dict={
        #                     self.image_features: feat})
        #             sentence = " ".join(
        #                 self.IDs_to_Words(
        #                     self.idxtow, caption_IDs))
        #             sentence = sentence.split("</S>")[0]
        #             print(sentence)
        # image = 'dogs.jpg'
        # features = get_features(self.sess, self.io, image, self.saveencoder)

        # for i, feat in enumerate(features):
        #    feat = np.reshape(feat, newshape=(1, OUTPUT_SIZE))
        #    caption_IDs = self.sess.run(
        #        self.IDs, feed_dict={
        #            self.image_features: feat})
        #    sentence = " ".join(
        #        self.IDs_to_Words(
        #            self.idxtow, caption_IDs))
        #    sentence = sentence.split("</S>")[0]
        #    print(sentence)

        # import matplotlib.pyplot as plt
        # import matplotlib.image as mpimg
        # image_printed = mpimg.imread('dogs.jpg')
        # plt.imshow(image_printed)
        # plt.axis("off")
        # plt.title(sentence, fontsize='13', loc='center')
        # plt.show()


        # if self.load_image:
        #     plt.imshow(Image.open(path))
        #     plt.axis("off")
        #     plt.title(sentence, fontsize='10', loc='left')
        #     name=path.split("/")[-1]
        #     plt.savefig("./results/"+"gen_"+name)
        #     plt.show()
        # else:
        #     print(sentence)
        #
        # if self.savedecoder:
        #     saver = tf.train.Saver()
        #     saver.save(self.sess, "model/Decoder/model.ckpt")
        # # #return path, sentence

    def batch_decoder(self, filenames, path,features):
        saver = tf.train.Saver()
        ckpt_file = "./model/model.ckpt-" + str(self.current_step)
        sentences = []
        filenames = np.unique(filenames)
        with open("model/Decoder/Generated_Captions.txt", 'w') as f:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                saver.restore(sess, ckpt_file)
                for i, feat in enumerate(features):
                    feat = np.reshape(feat, newshape=(1, OUTPUT_SIZE))
                    caption_IDs = sess.run(
                        self.IDs, feed_dict={
                            self.image_features: feat})
                    sentence = " ".join(
                        self.IDs_to_Words(
                            self.idxtow, caption_IDs))
                    sentence = sentence.split("</S>")[0]
                    if i % 1000 == 0:
                        print("Progress", i, "out of", features.shape[0])
                    f.write(filenames[i] + "\t" + sentence + "\n")
