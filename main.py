import numpy as np
import os
import sys
import math
from tqdm import tqdm
import argparse

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, EarlyStopping

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

import utils.models as mod
from utils.input_data import get_datasets, run_augmentation
import utils.datasets as ds
import utils.helper as hlp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs augmentation model.')
    # General settings
    parser.add_argument('--gpus', type=str, default="", help="Sets CUDA_VISIBLE_DEVICES")
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset to test (required, ex: unipen1a)')
    parser.add_argument('--model', type=str, default="vgg", help="Set model name")
    parser.add_argument('--train', default=False, action="store_true", help="Train?")
    parser.add_argument('--save', default=False, action="store_true", help="save to disk?")
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    # File settings
    parser.add_argument('--preset_files', default=False, action="store_true", help="Use preset files")
    parser.add_argument('--ucr', default=False, action="store_true", help="Use UCR 2015")
    parser.add_argument('--ucr2018', default=False, action="store_true", help="Use UCR 2018")
    parser.add_argument('--data_dir', type=str, default="data", help="Data dir")
    parser.add_argument('--train_data_file', type=str, default="", help="Train data file")
    parser.add_argument('--train_labels_file', type=str, default="", help="Train label file")
    parser.add_argument('--test_data_file', type=str, default="", help="Test data file")
    parser.add_argument('--test_labels_file', type=str, default="", help="Test label file")
    parser.add_argument('--test_split', type=int, default=0, help="test split")
    parser.add_argument('--weight_dir', type=str, default="weights", help="Weight path")
    parser.add_argument('--log_dir', type=str, default="logs", help="Log path")
    parser.add_argument('--output_dir', type=str, default="output", help="Output path")
    parser.add_argument('--normalize_input', default=False, action="store_true", help="Normalize between [-1,1]")
    parser.add_argument('--delimiter', type=str, default=" ", help="Delimiter")
    
    # Network settings
    parser.add_argument('--optimizer', type=str, default="sgd", help="Which optimizer")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
    parser.add_argument('--validation_split', type=int, default=0, help="size of validation set")
    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--verbose', type=int, default=1, help="verbose")
    
    args = parser.parse_args()
        
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  
    
    nb_class = ds.nb_classes(args.dataset)
    nb_dims = ds.nb_dims(args.dataset)
        
    # Load data
    x_train, y_train, x_test, y_test = get_datasets(args)
    
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)
        
    # Process data
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1])) 
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1])) 
    y_test = to_categorical(ds.class_offset(y_test, args.dataset), nb_class)
    y_train = to_categorical(ds.class_offset(y_train, args.dataset), nb_class)
    
    # Augment data
    x_train, y_train, augmentation_tags = run_augmentation(x_train, y_train, args)
    model_prefix = "%s_%s%s"%(args.model, args.dataset, augmentation_tags)
        
    nb_iterations = args.iterations
    batch_size = args.batch_size
    nb_epochs = np.ceil(nb_iterations * (batch_size / x_train.shape[0])).astype(int)
    
    model = mod.get_model(args.model, input_shape, nb_class)
    
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=np.ceil(nb_epochs/20.).astype(int), verbose=args.verbose, mode='auto', min_lr=1e-5, cooldown=np.ceil(nb_epochs/40.).astype(int))
    
    if args.save:
        if not os.path.exists(args.weight_dir):
            os.mkdir(args.weight_dir)
        if not os.path.exists(os.path.join(args.weight_dir, model_prefix)):
            os.mkdir(os.path.join(args.weight_dir, model_prefix))
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
    #     model_checkpoint = ModelCheckpoint(os.path.join(args.weight_dir, model_prefix, "%s_best_train_acc_weights.h5" % (model_prefix)), verbose=1, monitor='acc', save_best_only=True)
        if not os.path.exists(os.path.join(args.log_dir, model_prefix)):
            os.mkdir(os.path.join(args.log_dir, model_prefix))
        csv_logger = CSVLogger(os.path.join(args.log_dir, '%s.csv' % (model_prefix)))
        
        callback_list = [reduce_lr, csv_logger]
#         callback_list = [model_checkpoint, reduce_lr, csv_logger]
    else:
        callback_list = [reduce_lr]
    
    if args.optimizer=="adam":
        from tensorflow.keras.optimizers import Adam
        optm = Adam(lr=args.lr)
    elif args.optimizer=="nadam":
        from tensorflow.keras.optimizers import Nadam
        optm = Nadam(lr=args.lr)
    elif args.optimizer=="adadelta":
        from tensorflow.keras.optimizers import Adadelta
        optm = Adadelta(lr=args.lr, rho=0.95, epsilon=1e-8)
    else:
        from tensorflow.keras.optimizers import SGD
        optm = SGD(lr=args.lr, decay=5e-4, momentum=0.9) #, nesterov=True)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #train
    if args.train:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callback_list, verbose=args.verbose, validation_split=args.validation_split)
        if args.save:
            model.save_weights(os.path.join(args.weight_dir, model_prefix, "%s_final_weights.h5" % (model_prefix)))
    else:    
        model.load_weights(os.path.join(args.weight_dir, model_prefix, "%s_final_weights.h5" % (model_prefix)))
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test: ", accuracy)
    
    if args.save:
        y_preds = np.array(model.predict(x_test, batch_size=batch_size))
        y_preds = np.argmax(y_preds, axis=1)
        np.savetxt(os.path.join(args.output_dir, "%s_%.15f.txt" % (model_prefix,accuracy)), y_preds, fmt="%d")
    
    if os.path.exists(os.path.join(args.weight_dir, model_prefix, "%s_best_train_acc_weights.h5" % (model_prefix))):
        model.load_weights(os.path.join(args.weight_dir, model_prefix, "%s_best_train_acc_weights.h5" % (model_prefix)))
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
        print("Best Train Acc, Test: ", accuracy)