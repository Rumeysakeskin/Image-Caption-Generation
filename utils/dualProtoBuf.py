import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()


with open('model/Trained_Graphs/encoder_frozen_model.pb', 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='encoder', op_dict=None, producer_op_list=None)
graph = tf.get_default_graph()
tensors_encoder = [n.name for n in tf.get_default_graph().as_graph_def().node]

def init_encoder():
    sess = tf.Session()
    return sess

def encoder_forward_pass(sess, image_path):
    # for op in graph.get_operations():
    #     print(str(op.name))
    #
    # [n.name for n in tf.get_default_graph().as_graph_def().node]
    in1 = graph.get_tensor_by_name("encoder/InputFile:0")
    out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
    feed_dict = {in1: image_path}
    prepro_image = sess.run(out1, feed_dict=feed_dict)
    print(prepro_image)
    in2 = graph.get_tensor_by_name("encoder/import/InputImage:0")
    outfinal = graph.get_tensor_by_name("encoder/import/global_average_pooling2d_1/Mean:0")
    feed_dict = {in2: prepro_image}
    features = sess.run(outfinal, feed_dict=feed_dict)[0][0]
    return features
sess = init_encoder()

features = encoder_forward_pass(sess, "images/giraffe.jpg")


def IDs_to_Words(ID_batch):
    return [idxtow[word] for IDs in ID_batch for word in IDs]


tf.reset_default_graph()

with open('model/Trained_Graphs/decoder_frozen_model.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='decoder', op_dict=None,
                    producer_op_list=None)
graph = tf.get_default_graph()
tensors_decoder = [n.name for n in tf.get_default_graph().as_graph_def().node]

wtoidx = np.load("Dataset/wordmap.npy",allow_pickle=True).tolist()
idxtow = dict(zip(wtoidx.values(), wtoidx.keys()))

with open("model/Decoder/DecoderOutputs.txt", 'r') as fr:
    outputs = fr.read()
    outputs = outputs.split('\n')[:-1]


def init_decoder():
    sess = tf.Session()
    return sess


def decoder_forward_pass(sess, features):
    feed_dict = {graph.get_tensor_by_name("decoder/Input_Features:0"): features}
    prob_tensor = []
    for i, outs in enumerate(outputs):
        prob_tensor.append(graph.get_tensor_by_name("decoder/" + outs + ":0"))
    prob = sess.run(prob_tensor, feed_dict=feed_dict)
    return " ".join(IDs_to_Words(prob)).split("</S>")[0]
sess= init_decoder()
print(decoder_forward_pass(sess,features))