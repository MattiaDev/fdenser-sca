import logging
from collections import namedtuple

from sklearn.model_selection import train_test_split
import keras
import h5py
import numpy as np


logger = logging.getLogger(__name__)

SubSet = namedtuple('SubSet', ['x', 'y', 'meta'])


def load_ascad(ds_path):
    f = h5py.File(ds_path, 'r')

    profiling_traces = np.array(f['Profiling_traces']['traces'])[..., np.newaxis]
    profiling_labels = np.array(f['Profiling_traces']['labels'])
    profiling_metadata = np.array(f['Profiling_traces']['metadata'])

    x_final_test = np.array(f['Attack_traces']['traces'])[..., np.newaxis]
    y_final_test = np.array(f['Attack_traces']['labels'])
    meta_final_test = np.array(f['Attack_traces']['metadata'])

    total_train_size = len(profiling_traces)
    val_size = int(total_train_size * 0.1)
    test_size = int(total_train_size * 0.1)

    # Get evo val set
    (
        x_train_tmp,
        x_val,
        y_train_tmp,
        y_val,
        meta_train_tmp,
        meta_val,
    )  = train_test_split(
        profiling_traces,
        profiling_labels,
        profiling_metadata,
        test_size=val_size,
        stratify=profiling_labels,
        random_state=42,
    )

    # Get evo test set
    (
        x_train,
        x_test,
        y_train,
        y_test,
        meta_train,
        meta_test,
    )  = train_test_split(
        x_train_tmp,
        y_train_tmp,
        meta_train_tmp,
        test_size=test_size,
        stratify=y_train_tmp,
        random_state=42,
    )

    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    dataset = {
        'train': SubSet(x_train, y_train, meta_train),
        'val': SubSet(x_val, y_val, meta_val),
        'test': SubSet(x_test, y_test, meta_test),
        'final_test': SubSet(x_final_test, y_final_test, meta_final_test),
    }
    return dataset, dataset['train'].x[0].shape


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
        return load_ascad('ASCAD.h5')
    else:
        print('Error: the dataset is not valid')
        raise ValueError
