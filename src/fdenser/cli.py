import argparse
import os
import sys


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
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Based on FastDenser++\nAn example invocation could be:'
            '\n\nfastdenser -d fashion-mnist -c ../fdenser-sca/example/cpu.yml -r 11 -g ../fdenser-sca/example/cnn.grammar'
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

    from .engine import main
    from .execution import (
        load_config,
        load_dataset,
    )
    from .grammar import Grammar

    # load config file
    config = load_config(args.config, args.run)

    # load grammar
    grammar = Grammar(args.grammar)

    # load dataset
    evo_dataset, input_shape = load_dataset(args.dataset)

    # execute
    main(args.run, evo_dataset, input_shape, config, grammar)
