import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
import time
from keras.utils import to_categorical


from utils.input_data import get_datasets, run_augmentation
import utils.datasets as ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evalates average time for augmentation.')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=1, help="How many times to augment")
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
    parser.add_argument('--preset_files', default=True, action="store_true", help="Use preset files")
    parser.add_argument('--ucr', default=False, action="store_true", help="Use UCR 2015")
    parser.add_argument('--ucr2018', default=True, action="store_true", help="Use UCR 2018")
    parser.add_argument('--data_dir', type=str, default="data", help="Data dir")
    parser.add_argument('--train_data_file', type=str, default="", help="Train data file")
    parser.add_argument('--train_labels_file', type=str, default="", help="Train label file")
    parser.add_argument('--test_data_file', type=str, default="", help="Test data file")
    parser.add_argument('--test_labels_file', type=str, default="", help="Test label file")
    parser.add_argument('--test_split', type=int, default=0, help="test split")
    parser.add_argument('--weight_dir', type=str, default="weights", help="Weight path")
    parser.add_argument('--log_dir', type=str, default="logs", help="Log path")
    parser.add_argument('--output_dir', type=str, default="output", help="Output path")
    parser.add_argument('--normalize_input', default=True, action="store_true", help="Normalize between [-1,1]")
    parser.add_argument('--delimiter', type=str, default=" ", help="Delimiter")
    args = parser.parse_args()
    
    datasets = ["Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinCECGTorso", 
                "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", 
                "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", 
                "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", 
                "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", 
                "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", 
                "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", 
                "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect",
                "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", 
                "RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", 
                "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", 
                "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryAll", 
                "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", 
                "Worms", "WormsTwoClass", "Yoga", "ACSF1", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ", 
                "BME", "Chinatown", "Crop", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "EOGHorizontalSignal", 
                "EOGVerticalSignal", "EthanolLevel", "FreezerRegularTrain", "FreezerSmallTrain", "Fungi", "GestureMidAirD1", 
                "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPointAgeSpan", 
                "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "HouseTwenty", "InsectEPGRegularTrain", "InsectEPGSmallTrain", 
                "MelbournePedestrian", "MixedShapesRegularTrain", "MixedShapesSmallTrain", "PickupGestureWiimoteZ", 
                "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "PowerCons","Rock","SemgHandGenderCh2",
                "SemgHandMovementCh2","SemgHandSubjectCh2","ShakeGestureWiimoteZ","SmoothSubspace","UMD"]
    total = 0
    for i, dataset in enumerate(datasets):
        args.dataset = dataset
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
        start = time.time()
        x_train, y_train, augmentation_tags = run_augmentation(x_train, y_train, args)
        total += time.time()-start
    print(total)
    print(total/128.)