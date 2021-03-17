import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np
import argparse
import os

batch_size = 10
files, input_layer, output_layer = [None]*3

IMAGE_SIZE = 299 # for inception v4 - 299
OUTPUT_SIZE = 2048 # 1536

def build_prepro_graph(inception_path):

    global input_layer, output_layer
    with open(inception_path, 'rb') as f:
        fileContent = f.read()

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def)
    graph = tf.compat.v1.get_default_graph()

    for op in graph.get_operations():
        print(str(op.name))
        #[n.name for n in tf.get_default_graph().as_graph_def().node]

    input_layer = graph.get_tensor_by_name("import/input_1:0")
    output_layer = graph.get_tensor_by_name(
        "import/global_average_pooling2d_1/Mean:0")

    print("input_layer:")
    print(input_layer)
    print("output_layer:")
    print(output_layer)

    input_file = tf.placeholder(dtype=tf.string, name="InputFile")
    image_file = tf.io.read_file(input_file)
    print(image_file)
    jpg = tf.image.decode_jpeg(image_file, channels=3)
    png = tf.image.decode_png(image_file, channels=3)
    output_jpg =  tf.image.resize(jpg, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    output_jpg = tf.reshape(
        output_jpg, [
            1, IMAGE_SIZE, IMAGE_SIZE, 3], name="Preprocessed_JPG")
    output_png = tf.image.resize_images(png, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    output_png = tf.reshape(
        output_png, [
            1, IMAGE_SIZE, IMAGE_SIZE, 3], name="Preprocessed_PNG")
    return input_file, output_jpg, output_png


def load_image(sess, io, image):
    if image.split('.')[-1] == "png":
        return sess.run(io[2], feed_dict={io[0]: image})
    return sess.run(io[1], feed_dict={io[0]: image})


def load_next_batch(sess, io, img_path):
    for batch_idx in range(0, len(files), batch_size):
        batch = files[batch_idx:batch_idx + batch_size]
        shape = (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)
        batch = np.zeros(shape=shape, dtype=np.float32)
        # batch = np.array(
        #     list(map(lambda x: load_image(sess, io, img_path + x), batch)))
        # batch = batch.reshape((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))

        yield batch

def forward_pass(io, img_path):
    global output_layer, files
    files = sorted(np.array(os.listdir(img_path)))
    print("#Images:", len(files))
    n_batch = int(len(files) / batch_size)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter = load_next_batch(sess, io, img_path)
        for i in range(n_batch):
            batch = batch_iter.__next__()
            assert batch.shape == (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)
            feed_dict = {input_layer: batch}
            if i is 0:
                # shape = (len(files), OUTPUT_SIZE)
                prob = sess.run(
                    output_layer, feed_dict=feed_dict).reshape(
                    batch_size, OUTPUT_SIZE)

            else:
                prob = np.append(
                    prob,
                    sess.run(
                        output_layer,
                        feed_dict=feed_dict).reshape(
                        batch_size,
                        OUTPUT_SIZE),
                    axis=0)
            if i % 5 == 0:
                print("Progress:" + str(((i + 1) / float(n_batch) * 100)) + "%\n")
    print("Progress:" + str(((n_batch) / float(n_batch) * 100)) + "%\n")
    print("Saving Features : features.npy\n")
    np.save('Dataset/features', prob)


def get_features(sess, io, img, saveencoder=False):
    # print('get_features:img')
    # print(img)
    global output_layer
    output_layer = tf.reshape(output_layer, [1,OUTPUT_SIZE], name="Output_Features")
    image = load_image(sess, io, img)
    feed_dict = {input_layer: image}
    prob = sess.run(output_layer, feed_dict=feed_dict)

    if saveencoder:
        tensors = [n.name for n in sess.graph.as_graph_def().node]
        with open("model/Encoder/Encoder_Tensors.txt", 'w') as f:
            for t in tensors:
                f.write(t + "\n")
        saver = tf.train.Saver()
        saver.save(sess, "model/Encoder/model.ckpt")
    return prob

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path"
        "",
        type=str,
        help="A valid path to MSCCOCO/flickr30k images(unzipped)",
        required=True)
    parser.add_argument(
        "--inception_path",
        type=str,
        help="A valid path to inception_v4.pb",
        required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_arguments()

    print("Extracting Features")
    io = build_prepro_graph(args.inception_path)

    forward_pass(io, args.data_path)
    print("done")
