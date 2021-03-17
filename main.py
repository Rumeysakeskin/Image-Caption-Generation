import joblib

from caption_generator import *
from utils.data_util import generate_captions
from configuration import Configuration
import os, sys
import argparse
import json
from utils.data_util import *
import pickle, gzip

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="train|test|eval",
    choices=[
        "train",
        "test",
        "eval"],
    required=True)
parser.add_argument(
    "--resume",
    help="make model training resumable",
    action="store_true")
parser.add_argument(
    "--caption_path",
    type=str,
    help="A valid path to COCO/flickr30k caption file: results_20130124.token/captions_val2014.json")
parser.add_argument(
    "--feature_path",
    type=str,
    help="A valid path to COCO/flickr30k image features: features.npy")
parser.add_argument(
    "--data_is_coco",
    help="Is dataset MSCOCO? converts COCO caption data to flickr30k format",
    action="store_true")
parser.add_argument(
    "--inception_path",
    type=str,
    help="A valid path to inception_v4.pb",
    default="ConvNets/InceptionV3_keras.pb")
parser.add_argument(
    "--saveencoder",
    help="Save Decoder graph in model/Encoder/",
    action="store_true")
parser.add_argument(
    "--savedecoder",
    help="Save Decoder graph in model/Decoder/",
    action="store_true")
parser.add_argument(
    "--image_path",
    type=str,
    help="If mode is test then, Path to the Image for Generation of Captions")
parser.add_argument(
    "--load_image",
    help="If mode is test then, displays and stores image with generated caption",
    action="store_true")
parser.add_argument(
    "--validation_data",
    type=str,
    help="If mode is eval then, Path to the Validation Data for evaluation")
args = parser.parse_args()
config = Configuration(vars(args))

if config.mode == "train":
    vocab, wtoidx, training_data = generate_captions(
        config.word_threshold, config.max_len, args.caption_path, args.feature_path,
        config.data_is_coco)
    features, captions = training_data[:, 0], training_data[:, 1]
    features = np.array([feat.astype(float) for feat in features])
    data = (vocab.tolist(), wtoidx.tolist(), features, captions)
    model = Caption_Generator(config, data=data)
    loss, inp_dict = model.build_train_graph()
    model.train(loss, inp_dict)

elif config.mode == "test":
    if os.path.exists(args.image_path):
        model = Caption_Generator(config)
        model.decode(args.image_path)
    else:
        print("Please provide a valid image path.\n Usage:\n python main.py --mode test --image_path VALID_PATH")

elif config.mode == "eval":
    config.mode = "test"
    # config.batch_decode = True

    if os.path.exists(args.image_path):
        model = Caption_Generator(config)
        model.decode(args.image_path)

        # features = np.load(args.validation_data,  allow_pickle=True)
        # # with open("Dataset/Validation_Captions.txt") as f:
        # #    data = f.readlines()
        #
        # with open("Dataset/captions_val2017.json", 'r') as f:
        #     data = json.load(f)
        #
        # # filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
        # filenames = sorted([d["file_name"].split('.')[0] for d in data['images']])
        # # captions = [caps.replace('\n', '').split('\t')[1] for caps in data]
        # # features, captions = validation_data[:, 0], validation_data[:, 1]
        # features = np.array([feat.astype(float) for feat in features])
        # model = Caption_Generator(config)
        # generated_captions = model.batch_decoder(filenames,args.image_path,features)


