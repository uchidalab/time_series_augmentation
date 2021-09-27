import numpy as np
import os

def load_data_from_file(data_file, label_file=None, delimiter=" "):
    if label_file:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = np.genfromtxt(label_file, delimiter=delimiter)
        if labels.ndim > 1:
            labels = labels[:,1]
    else:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = data[:,0]
        data = data[:,1:]
    return data, labels
    
def read_data_sets(train_file, train_label=None, test_file=None, test_label=None, test_split=0.1, delimiter=" "):
    train_data, train_labels = load_data_from_file(train_file, train_label, delimiter)
    if test_file:
        test_data, test_labels = load_data_from_file(test_file, test_label, delimiter)
    else:
        test_size = int(test_split * float(train_labels.shape[0]))
        test_data = train_data[:test_size]
        test_labels = train_labels[:test_size]
        train_data = train_data[test_size:]
        train_labels = train_labels[test_size:]
    return train_data, train_labels, test_data, test_labels


def get_datasets(args):
    # Load data
    if args.preset_files:
        if args.ucr:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter=",")
        elif args.ucr2018:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN.tsv"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST.tsv"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
        else:
            x_train_file = os.path.join(args.data_dir, "train-%s-data.txt"%(args.dataset))
            y_train_file = os.path.join(args.data_dir, "train-%s-labels.txt"%(args.dataset))
            x_test_file = os.path.join(args.data_dir, "test-%s-data.txt"%(args.dataset))
            y_test_file = os.path.join(args.data_dir, "test-%s-labels.txt"%(args.dataset))
            x_train, y_train, x_test, y_test = read_data_sets(x_train_file, y_train_file, x_test_file, y_test_file, test_split=args.test_split, delimiter=args.delimiter)
    else:
        x_train, y_train, x_test, y_test = read_data_sets(args.train_data_file, args.train_labels_file, args.test_data_file, args.test_labels_file, test_split=args.test_split, delimiter=args.delimiter)
    
    # Normalize
    if args.normalize_input:
        x_train_max = np.nanmax(x_train)
        x_train_min = np.nanmin(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
        
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    return x_train, y_train, x_test, y_test

def run_augmentation(x, y, args):
    print("Augmenting %s"%args.dataset)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print("Round %d: %s done"%(n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag
    return x_aug, y_aug, augmentation_tags

def augment(x, y, args):
    import utils.augmentation as aug
    augmentation_tags = ""
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = aug.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags
