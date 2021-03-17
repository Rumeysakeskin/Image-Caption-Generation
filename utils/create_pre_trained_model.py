import tensorflow as tf

graph_file = "nasnet_large_graphdef.pb"
graph_def = tf.GraphDef()
with open(graph_file, "rb") as f:
  graph_def.ParseFromString(f.read())

for node in graph_def.node:
  print(node.name)