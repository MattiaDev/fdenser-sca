import argparse
import os
import sys

from .engine import main
from .execution import (
    load_config,
    load_dataset,
)
from .grammar import Grammar


def fast_denser_cli():
    """
        Maps and checks the input parameters and call the main function.

        Parameters
        ----------
        argv : list
            argv from system
    """

    parser = argparse.ArgumentParser(
        prog='fastdenser',
        description='A neuroevolution engine to evolve CNNs',
        epilog='Based on FastDenser++'
    )
    parser.add_argument("-c", "--config", help="YAML config file for the engine", required=True)
    parser.add_argument("-d", "--dataset", help="The name of the built-in dataset to load", required=True)
    parser.add_argument("-g", "--grammar", help="The grammar file", required=True)
    parser.add_argument("-r", "--run", help="The number of the current run", default=0, type=int)

    args = parser.parse_args()

    # check if files exist
    if not os.path.isfile(args.grammar):
        print('Grammar file does not exist.')
        sys.exit(-1)

    if not os.path.isfile(args.config):
        print('Configuration file does not exist.')
        sys.exit(-1)

    # load config file
    config = load_config(args.config)

    # load grammar
    grammar = Grammar(args.grammar)

    # load dataset
    evo_dataset = load_dataset(args.dataset)

    # execute
    main(args.run, evo_dataset, config, grammar)