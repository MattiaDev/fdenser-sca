from sklearn.model_selection import train_test_split
import keras
from multiprocessing import Pool
import tensorflow as tf
import contextlib
import sys
import h5py
import numpy as np


def prepare_data(x_train, y_train, x_test, y_test, n_classes):
    """
        Split the data into independent sets

        Parameters
        ----------
        x_train : np.array
            training instances
        y_train : np.array
            training labels
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels


        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of
                                          the model on unseen data
    """

    total_train_size = len(x_train)
    val_size = int(total_train_size * 0.05)
    test_size = int(total_train_size * 0.05)


    x_train, evo_x_val, y_train, evo_y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=y_train)

    evo_x_train, evo_x_test, evo_y_train, evo_y_test = train_test_split(
        x_train, y_train, test_size=test_size, stratify=y_train)

    evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
    evo_y_val = keras.utils.to_categorical(evo_y_val, n_classes)

    dataset = {
        'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
        'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
        'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
        'x_test': x_test, 'y_test': y_test
    }

    print('DS Shape', evo_x_train.shape)
    print('Item Shape', evo_x_train[0].shape)

    return dataset, evo_x_train[0].shape


def load_dataset(dataset):
    """
        Load a specific dataset

        Parameters
        ----------
        dataset : str
            dataset to load

        shape : tuple(int, int)
            shape of the instances

        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of
                                          the model on unseen data
    """

    if dataset == 'ascad':
        f = h5py.File('ASCAD.h5', 'r')

        profiling_traces = np.array(f['Profiling_traces']['traces'])[..., np.newaxis]
        profiling_labels = f['Profiling_traces']['labels']
        profiling_labels = keras.utils.to_categorical(profiling_labels)

        print(type(profiling_traces))
        print(type(profiling_labels))
        print(profiling_traces.shape)
        print(profiling_labels.shape)
        
        attack_traces = np.array(f['Attack_traces']['traces'])[..., np.newaxis]
        attack_labels = f['Attack_traces']['labels']
        attack_labels = keras.utils.to_categorical(attack_labels)

        print(type(attack_traces))
        print(type(attack_labels))
        print(attack_traces.shape)
        print(attack_labels.shape)

        x_train = profiling_traces
        y_train = profiling_labels
        x_test = attack_traces
        y_test = attack_labels

        n_classes = 255
    else:
        print('Error: the dataset is not valid')
        sys.exit(-1)

    dataset, input_shape = prepare_data(x_train, y_train, x_test, y_test, n_classes)

    return dataset, input_shape
