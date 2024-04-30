import os
from pathlib import Path


class EvolutionaryConfig:
    def __init__(self, evolutionary_dict):
        self.max_generations = evolutionary_dict['num_generations']
        self.max_epochs = evolutionary_dict['max_epochs']
        self.es_lambda = evolutionary_dict['lambda']
        self.default_train_time = evolutionary_dict['default_train_time']
        self.minimize = evolutionary_dict['minimize']
        self.fitness = evolutionary_dict['fitness_function']
        self.mutations = evolutionary_dict['mutations']

    def __str__(self):
        desc = 'Evolutionary Configuration:\n'
        desc += f'- max generation: {self.max_generations}\n'
        desc += f'- max epochs: {self.max_epochs}\n'
        desc += f'- lambda: {self.es_lambda}\n'
        desc += f'- default train time: {self.default_train_time}\n'
        desc += f'- is a minimization problem: {self.minimize}\n'
        desc += f'- fitness function: {self.fitness.__name__}\n'
        desc += f'- mutation rates:\n'
        for k,v in self.mutations.items():
            desc += f'  * {k}: {v}\n'
        return desc


class NetworkConfig:
    def __init__(self, network_dict):
        self.structure = network_dict['network_structure']
        self.initial = network_dict['network_structure_init']
        self.macro_structure = network_dict['macro_structure']
        self.output = network_dict['output']
        self.levels_back = network_dict['levels_back']

    def __str__(self):
        desc = 'Network Configuration:\n'
        desc += f'- network structure: {self.structure}\n'
        desc += f'- initial network structure: {self.initial}\n'
        desc += f'- macro structure: {self.macro_structure}\n'
        desc += f'- output: {self.output}\n'
        desc += f'- levels back: {self.levels_back}\n'
        return desc


class SetupConfig:
    def __init__(self, setup_dict, run):
        self.random_seeds = setup_dict['random_seeds']
        self.numpy_seeds = setup_dict['numpy_seeds']
        self.path = Path(setup_dict['save_path'])
        self.run_path = self.path / Path(f'run_{run:02d}')
        self.log_path = self.run_path / Path('fdenser.log')
        os.makedirs(self.run_path, exist_ok=True)

    def __str__(self):
        desc = 'Setup Configuration:\n'
        desc += f'- random seeds: {self.random_seeds}\n'
        desc += f'- numpy seeds: {self.numpy_seeds}\n'
        desc += f'- save path: {self.path}\n'
        desc += f'- run path: {self.run_path}\n'
        desc += f'- log path: {self.log_path}\n'
        return desc


class Config:
    def __init__(self, config, run):
        self.evo = EvolutionaryConfig(config['evolutionary'])
        self.network = NetworkConfig(config['network'])
        self.setup = SetupConfig(config['setup'], run)

    def __str__(self):
        desc = 'Configuration:\n'
        desc += str(self.evo)
        desc += str(self.network)
        desc += str(self.setup)
        return desc
