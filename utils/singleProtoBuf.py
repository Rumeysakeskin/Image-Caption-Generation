import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
with open('model/Trained_Graphs/merged_frozen_graph.pb', 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='', op_dict=None, producer_op_list=None)
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
wtoidx = np.load("Dataset/wordmap.npy",allow_pickle=True).tolist()
idxtow = dict(zip(wtoidx.values(), wtoidx.keys()))

with open("model/IdmapInceptionV3_RNN", 'w') as f:
    for value in idxtow.values():
        f.write(value + "\n")
print("Idmap is created")

# with open('model/Decoder/DecoderOutputs.txt', 'r') as fr:
#     outputs= fr.read()
#     outputs=outputs.split('\n')[:-1]
#     print(outputs)
# def IDs_to_Words(ID_batch):
#     return [idxtow[word] for IDs in ID_batch for word in IDs]
#
# def load_image(path, caption):
#     plt.imshow(Image.open(path))
#     plt.axis("off")
#     plt.title(caption, fontsize='10', loc='left')
#     arr=path.split("/")
#     plt.savefig("images"+arr[-1].split('.')[0]+".png")
#     plt.show()
#
#
# in1 = None
# out1 = None
# in2 = None
# sentence = None
#
#
# def get_tensors():
#     global in1, out1, in2, sentence
#     in1 = graph.get_tensor_by_name("encoder/InputFile:0")
#     out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
#     in2 = graph.get_tensor_by_name("encoder/import/InputImage:0")
#     sentence = []
#     for i, outs in enumerate(outputs):
#         sentence.append(graph.get_tensor_by_name("decoder/" + outs + ":0"))
#
#
# def init_caption_generator():
#     sess = tf.Session()
#     get_tensors()
#     return sess
#
#
# def preprocess_image(sess, image_path):
#     global in1, out1
#     if image_path.split(".")[-1] == "png":
#         out1 = graph.get_tensor_by_name("encoder/Preprocessed_PNG:0")
#     feed_dict = {in1: image_path}
#     prepro_image = sess.run(out1, feed_dict=feed_dict)
#     return prepro_image
#
#
# def generate_caption(sess, image_path):
#     global in2, out1, sentence
#
#     prepro_image = preprocess_image(sess, image_path)
#
#     feed_dict = {in2: prepro_image}
#     prob = sess.run(sentence, feed_dict=feed_dict)
#     # set default back to JPG
#     out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
#     caption = " ".join(IDs_to_Words(prob)).split("</S>")[0]
#     load_image(image_path, caption)
#
# sess = init_caption_generator()
# path="images/"
# files=sorted(os.listdir(path))
# files=[path+f for f in files]
# for f in files:
#     if os.path.splitext(f)[1] in [".png",".jpg",".jpeg"]:
#         generate_caption(sess, f)