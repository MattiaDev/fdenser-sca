import contextlib
import logging
from multiprocessing import Pool
from time import time

import keras
import numpy as np

from .module import Module


logger = logging.getLogger(__name__)


class Individual:
    """
        Candidate solution.


        Attributes
        ----------
        network_structure : list
            ordered list of tuples formated as follows
            [(non-terminal, min_expansions, max_expansions), ...]

        output_rule : str
            output non-terminal symbol

        macro_rules : list
            list of non-terminals (str) with the marco rules (e.g., learning)

        modules : list
            list of Modules (genotype) of the layers

        output : dict
            output rule genotype

        macro : list
            list of Modules (genotype) for the macro rules

        phenotype : str
            phenotype of the candidate solution

        fitness : float
            fitness value of the candidate solution

        metrics : dict
            training metrics

        num_epochs : int
            number of performed epochs during training

        trainable_parameters : int
            number of trainable parameters of the network

        time : float
            network training time

        current_time : float
            performed network training time

        train_time : float
            maximum training time

        id : int
            individual unique identifier


        Methods
        -------
            initialise(grammar, levels_back, reuse)
                Randomly creates a candidate solution

            decode(grammar)
                Maps the genotype to the phenotype

            evaluate(grammar, cnn_eval, weights_save_path,
                     parent_weights_path='')
                Performs the evaluation of a candidate solution
    """

    def __init__(self, network_structure, macro_rules, input_shape, output_rule, ind_id):
        """
            Parameters
            ----------
            network_structure : list
                ordered list of tuples formated as follows
                [(non-terminal, min_expansions, max_expansions), ...]

            macro_rules : list
                list of non-terminals (str) with the marco rules
                (e.g., learning)

            output_rule : str
                output non-terminal symbol

            ind_id : int
                individual unique identifier
        """

        self.network_structure = network_structure
        self.input_shape = input_shape
        self.output_rule = output_rule
        self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = 0
        self.trainable_parameters = None
        self.time = None
        self.current_time = 0
        self.train_time = 0
        self.id = ind_id

    def initialise(self, grammar, levels_back, reuse, init_max):
        """
            Randomly creates a candidate solution

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            levels_back : dict
                number of previous layers a given layer can receive as input

            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            candidate_solution : Individual
                randomly created candidate solution
        """

        for non_terminal, min_expansions, max_expansions in self.network_structure:
            new_module = Module(
                non_terminal,
                min_expansions,
                max_expansions,
                levels_back[non_terminal],
            )
            new_module.initialise(grammar, reuse, init_max)

            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self

    def decode(self, grammar):
        """
            Maps the genotype to the phenotype

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            Returns
            -------
            phenotype : str
                phenotype of the individual to be used in the mapping to the
                keras model.
        """

        phenotype = ''
        offset = 0
        layer_counter = 0
        for module in self.modules:
            offset = layer_counter
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter += 1
                phenotype += ' ' \
                    + grammar.decode(module.module, layer_genotype) \
                    + ' input:' \
                    + ",".join(map(str, np.array(module.connections[layer_idx])
                               + offset))

        phenotype += ' ' + grammar.decode(self.output_rule, self.output) \
            + ' input:' + str(layer_counter-1)

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += ' '+grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype

    def evaluate(self, grammar, cnn_eval, weights_save_path,
                 parent_weights_path=''):
        """
            Performs the evaluation of a candidate solution

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            cnn_eval : Evaluator
                Evaluator instance used to train the networks

            datagen : keras.preprocessing.image.ImageDataGenerator
                Data augmentation method image data generator

            weights_save_path : str
                path where to save the model weights after training

            parent_weights_path : str
                path to the weights of the previous training


            Returns
            -------
            fitness : float
                quality of the candidate solutions
        """

        phenotype = self.decode(grammar)
        start = time()

        load_prev_weights = True
        # Unless our mutation changed the training time only (thereby
        # leaving the structure untouched), then we will reload
        # previous weights
        if self.current_time == 0:
            # otherwise the mutation of the structure will reset the
            # current_time field to 0 and we won't load any previous
            # weights
            logger.debug('Current time is 0: training will start from scratch')
            load_prev_weights = False

        train_time = self.train_time - self.current_time

        num_pool_workers = 1
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(
                tf_evaluate,
                [(cnn_eval, phenotype, load_prev_weights,
                  weights_save_path, parent_weights_path,
                  train_time, self.num_epochs, self.input_shape)]
            )
            metrics = pool_results.get()[0]

        if metrics is not None:
            if 'val_accuracy' in metrics:
                if type(metrics['val_accuracy']) is list:
                    metrics['val_accuracy'] = [
                        i for i in metrics['val_accuracy']
                    ]
                else:
                    metrics['val_accuracy'] = [
                        i.item() for i in metrics['val_accuracy']
                    ]
            if 'loss' in metrics:
                if type(metrics['loss']) is list:
                    metrics['loss'] = [i for i in metrics['loss']]
                else:
                    metrics['loss'] = [i.item() for i in metrics['loss']]
            if 'accuracy' in metrics:
                if type(metrics['accuracy']) is list:
                    metrics['accuracy'] = [i for i in metrics['accuracy']]
                else:
                    metrics['accuracy'] = [
                        i.item() for i in metrics['accuracy']
                    ]
            self.metrics = metrics
            if 'accuracy_test' in metrics:
                if self.metrics['accuracy_test'] is None:
                    self.fitness = float('nan')
                    logger.warning(f'Failed to evaluate individual {self.id}')
                elif type(self.metrics['accuracy_test']) is float:
                    self.fitness = self.metrics['accuracy_test']
                elif type(self.metrics['accuracy_test']) is int:
                    self.fitness = self.metrics['accuracy_test']
                else:
                    self.fitness = self.metrics['accuracy_test'].item()
            if 'val_accuracy' in metrics:
                self.num_epochs += len(self.metrics['val_accuracy'])
            else:
                self.num_epochs += 1
            self.trainable_parameters = self.metrics['trainable_parameters']
            self.current_time += (self.train_time-self.current_time)
        else:
            logger.warning(f'Failed to train individual {self.id}')
            self.metrics = None
            self.fitness = float('nan')
            self.num_epochs = 0
            self.trainable_parameters = -1
            self.current_time = 0

        self.time = time() - start

        logger.info(f'Fitness of {self.id}: {self.fitness}')
        return self.fitness


def tf_evaluate(args):
    """
        Function used to deploy a new process to train a candidate solution.
        Each candidate solution is trained in a separe process to avoid
        memory problems.

        Parameters
        ----------
        args : tuple
            cnn_eval : Evaluator
                network evaluator

            phenotype : str
                individual phenotype

            load_prev_weights : bool
                resume training from a previous train or not

            weights_save_path : str
                path where to save the model weights after training

            parent_weights_path : str
                path to the weights of the previous training

            train_time : float
                maximum training time

            num_epochs : int
                maximum number of epochs

        Returns
        -------
        score_history : dict
            training data: loss and accuracy
    """

    import traceback
    import tensorflow as tf

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except IndexError:
        logger.debug('Evaluation will run on CPU')

    cnn_eval, phenotype, load_prev_weights, weights_save_path, \
        parent_weights_path, train_time, num_epochs, input_shape = args

    try:
        return cnn_eval.evaluate(
            phenotype,
            load_prev_weights,
            weights_save_path,
            parent_weights_path,
            train_time,
            num_epochs,
            input_shape,
        )
    except tf.errors.ResourceExhaustedError:
        logger.warning('Memory Error: ResourceExhaustedError')
        keras.backend.clear_session()
        return None
    except TypeError:
        logger.warning('Memory Error: TypeError')
        logger.debug(traceback.format_exc())
        keras.backend.clear_session()
        return None
