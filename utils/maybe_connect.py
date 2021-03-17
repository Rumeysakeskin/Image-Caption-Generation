import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util
encpb="model/Trained_Graphs/encoder_frozen_model.pb"
with open(encpb, 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
output1="Output_Features:0"
output1 = tf.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=[output1],
    name='encoder')
graph = tf.get_default_graph()
print("***")
print(output1)
decpb = "model/Trained_Graphs/decoder_frozen_model.pb"
with open(decpb, 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
input2="Input_Features:0"

tf.import_graph_def(
    graph_def,
    input_map={input2:output1[0]},
    return_elements=None,
    name='decoder')
graph = tf.get_default_graph()
with open("model/Decoder/DecoderOutputs.txt", 'r') as f:
    output = f.read()
    prefix = ""
    outputs=[]
    outputs += [prefix + o for o in output.split('\n')[:-1]]
outputs
import numpy as np
wtoidx = np.load("Dataset/wordmap.npy",allow_pickle=True).tolist()
idxtow = dict(zip(wtoidx.values(), wtoidx.keys()))
print(wtoidx.keys)
def IDs_to_Words(ID_batch):

    return [idxtow[word] for IDs in ID_batch for word in IDs]
with tf.Session() as sess:
    in1 = graph.get_tensor_by_name("encoder/InputFile:0")
    out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
    feed_dict = {in1: "images/ocean.jpg"}
    prepro_image = sess.run(out1, feed_dict=feed_dict)
    in2 = graph.get_tensor_by_name("encoder/import/InputImage:0")
    sentence = []
    for i,outs in enumerate(outputs):
        sentence.append(graph.get_tensor_by_name("decoder/"+outs+":0"))
    feed_dict = {in2:prepro_image}
    prob= sess.run(sentence, feed_dict=feed_dict)
    print (" ".join(IDs_to_Words(prob)).split("</S>")[0])
