import keras
import json
import sys
import numpy as np
import argparse
import os
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras import backend as K

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

#model = InceptionV3(weights='imagenet', input_tensor = Input(shape=(299, 299, 3)))
K.set_learning_phase(False)
model = keras.applications.InceptionV3(weights='imagenet', input_tensor = Input(shape=(299, 299, 3)),include_top=False)
model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
model.summary()
frozen_graph = freeze_session(K.get_session(),
                                 output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "ConvNets/", "InceptionV3_keras.pb", as_text=False)

print('input layer tensor:')
for t in model.inputs:
    print(t.name)

print('output layer tensor:')
for t in model.outputs:
    print(t.name)
print("they'll be use in convfeatures.py")
print("CNN model is created!")