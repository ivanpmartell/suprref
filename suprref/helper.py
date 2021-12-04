import os
import sys
import json
import torch
import random
import inspect
import requests
import argparse
import importlib
import numpy as np
from pathlib import Path
from itertools import product
from tqdm import tqdm
from PyInquirer import prompt, Separator, Validator, ValidationError

DATA_FOLDER = 'data/'

class FastaValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        data_folder = os.path.abspath(DATA_FOLDER)
        my_file = Path(text)
        if not my_file.is_file():
            raise ValidationError(
                message='Please enter a valid file',
                cursor_position=len(text))
        elif my_file.suffix not in ['.fa', '.fna', '.fsa', '.fasta']:
            raise ValidationError(
                message='Please enter a FASTA file. Extensions: .fa .fna .fsa .fasta',
                cursor_position=len(text))
        elif not str(os.path.dirname(my_file.absolute())) == data_folder:
            raise ValidationError(
                message='Make sure your data is in the correct folder',
                cursor_position=len(text)
                )


class AnnotationValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        data_folder = os.path.abspath(DATA_FOLDER)
        my_file = Path(text)
        if not my_file.is_file():
            raise ValidationError(
                message='Please enter a valid file',
                cursor_position=len(text))
        elif my_file.suffix not in ['.sga', '.bed']:
            raise ValidationError(
                message='Please enter an annotation file. Extensions: .sga and .bed',
                cursor_position=len(text))
        elif not str(os.path.dirname(my_file.absolute())) == data_folder:
            raise ValidationError(
                message='Make sure your data is in the correct folder',
                cursor_position=len(text)
                )


class FolderValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        my_folder = Path(text)
        if my_folder.exists():
            if not my_folder.is_dir():
                raise ValidationError(
                    message='Please enter a valid folder',
                    cursor_position=len(text))
        else:
            os.makedirs(my_folder)

class IsFileValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        my_file = Path(text)
        if not my_file.exists():
            raise ValidationError(
                    message='Please enter a file that exists',
                    cursor_position=len(text))


class FileExistsValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        my_file = Path(text)
        if my_file.exists():
            raise ValidationError(
                    message='Please enter a file that does not exist',
                    cursor_position=len(text))


class ConfigValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        config = Path(text)
        if config.is_file():
            if not config.suffix == '.json':
                raise ValidationError(
                    message='Please enter a configuration file. Extension: .json',
                    cursor_position=len(text))
        else:
            raise ValidationError(
                message='Please enter a valid file',
                cursor_position=len(text))


class PositiveValidator(Validator):
    def validate(self, document):
        if(type(document) == int):
            text = str(document)
        elif(type(document) == float):
            text = str(document)
        elif(type(document) == str):
            text = document
        else:
            text = document.text
        try:
            if not float(text) > 0:
                raise ValidationError(
                    message='Please enter a positive number',
                    cursor_position=len(text))
        except:
            raise ValidationError(
                message='Input %s was not a valid number' % text,
                cursor_position=len(text))


class SeedValidator(Validator):
    def validate(self, document):
        if(type(document) == int):
            text = str(document)
        elif(type(document) == str):
            text = document
        else:
            text = document.text
        try:
            if not int(text) >= 0:
                raise ValidationError(
                    message='Please enter a positive number or use 0 to generate random seed',
                    cursor_position=len(text))
        except:
            raise ValidationError(
                message='Seed was not a valid number',
                cursor_position=len(text))

class FunctionValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        if not text in Helper._NN_OUTPUT_FUNCTIONS:
            raise ValidationError(
                message=f'Please enter a valid output function. Functions: {Helper._NN_OUTPUT_FUNCTIONS}',
                cursor_position=len(text))

class ModuleValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        if not text in Helper._NN_MODULES:
            raise ValidationError(
                message=f'Please enter a valid module. Functions: {Helper._NN_MODULES}',
                cursor_position=len(text))

class OptimizerValidator(Validator):
    def validate(self, document):
        if(type(document) == str):
            text = document
        else:
            text = document.text
        if not text in Helper._NN_OPTIMIZERS:
            raise ValidationError(
                message=f'Please enter a valid optimizer. Functions: {Helper._NN_OPTIMIZERS}',
                cursor_position=len(text))

class Helper:
    DNA_DICT = {}
    _k = 1
    CONF_DICT: dict
    _FILE_TYPES = ['sld', 'fasta']
    _EXPERIMENT_CONFIGURATION_FILENAME = 'ExperimentConfiguration.json'
    _EXPERIMENT_MODULE_ARGS_FILENAME = 'module.args'
    _EXPERIMENT_OPTIMIZER_ARGS_FILENAME = 'optimizer.args'
    _NN_OUTPUT_FUNCTIONS = ['sigmoid', 'softmax']
    _NN_MODULES: dict
    _NN_MODULE_ARGS = {}
    _NN_OPTIMIZER_ARGS = {}
    _NN_OPTIMIZERS = ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD'] # Algorithms: https://pytorch.org/docs/stable/optim.html
    _F5_SPECIES = ['human', 'mouse', 'rat', 'dog', 'chicken', 'rhesus']
    _EPD_DATABASES = ['human', 'human_nc', 'M_mulatta', 'mouse', 'mouse_nc', 'R_norvegicus', 'C_familiaris', 'G_gallus', 'drosophila', 'A_mellifera', 'zebrafish', 'worm', 'arabidopsis', 'Z_mays', 'S_cerevisiae', 'S_pombe', 'P_falciparum']
    _EPD_TATA_FILTERS = ['all', 'with', 'without']
    _DATA_FOLDER = 'data/'
    _DICTKEY_NN_OUTPUT_FUNCTION = 'function'
    _DICTKEY_NN_MODULE = 'module'
    _DICTKEY_NN_TRAIN_ARGS = 'training_arguments'
    _DICTKEY_NN_MAX_EPOCHS = 'max_epochs'
    _DICTKEY_NN_PATIENCE = 'patience'
    _DICTKEY_NN_LEARNING_RATE = 'learning_rate'
    _DICTKEY_NN_BATCH_SIZE = 'batch_size'
    _DICTKEY_NN_OPTIMIZER = 'optimizer'
    _DICTKEY_INPUT_FILE = 'input'
    _DICTKEY_OUTPUT_FILE = 'output'
    _DICTKEY_ANNOTATIONS = 'annotations'
    _DICTKEY_CONFIGURATION = 'dataset_configuration'
    _DICTKEY_DATASET_TYPE = 'dataset'
    _DICTKEY_EXPERIMENT_FOLDER = 'experiment'
    _DICTKEY_SEQ_UPSTREAM_LEN = 'sequence_upstream_length'
    _DICTKEY_SEQ_DOWNSTREAM_LEN = 'sequence_downstream_length'
    _DICTKEY_STRIDE = 'stride'
    _DICTKEY_ERROR_TYPE = 'error_type'
    _DICTKEY_ERROR_MARGIN = 'error_margin'
    _DICTKEY_SEED = 'seed'
    _DICTKEY_F5_SPECIES = 'species'
    _DICTKEY_EPD_DATABASE = 'database'
    _DICTKEY_EPD_TATA_FILTER = 'tata_filter'
    _DICTKEY_INPUT_TYPE = 'input_type'
    _DICTKEY_OUTPUT_TYPE = 'output_type'
    _overwrite = False
    _maxepochs_default = 100
    _batchsize_default = 16
    _learningrate_default = 0.001
    _patience_default = 10
    _upstream_default = 1000 # promoter upstream length for the annotations
    _downstream_default = 400 # promoter downstream length for the annotations
    _stride_default = 50 # step size for the number of nucleotides to skip while moving the window
    _error_type_default = 'tss-proximity' # error types: proximity to TSS, sequence overlap
    _IO_ERROR_TYPES = ['sequence-overlap', 'tss-proximity', 'seq-proximity']
    _error_margin_default = [100] # list, if 1 value then its TSS proximity, if 2 values then its sequence overlap (from upstream, from downstream)
    _DATASET_CHOICES = ['IO', 'BIO', 'BME', 'CBG', 'CBPS', 'LITERATURE'] # "CBG" cluster by gene, "CBPS" cluster by promoter similarity

    def __init__(self):
        self.create_kmers()
        self._NN_MODULES = self.get_modules()
        dataset_configuration = {self._DICTKEY_SEQ_UPSTREAM_LEN: self._upstream_default,
                                 self._DICTKEY_SEQ_DOWNSTREAM_LEN: self._downstream_default,
                                 self._DICTKEY_STRIDE: self._stride_default,
                                 self._DICTKEY_SEED: 0}
        self.CONF_DICT = {self._DICTKEY_CONFIGURATION: dataset_configuration}

    def create_kmers(self):
        bases=['A','T','G','C']
        for i, p in enumerate(product(bases, repeat=self._k)):
            self.DNA_DICT[''.join(p)] = i

    def load_experiment(self):
        experiment_folder = self.CONF_DICT[self._DICTKEY_EXPERIMENT_FOLDER]
        config_file = os.path.join(experiment_folder, self._EXPERIMENT_CONFIGURATION_FILENAME)
        with open(config_file) as f:
            self.CONF_DICT = json.load(f)
        module_args_file = os.path.join(experiment_folder, self._EXPERIMENT_MODULE_ARGS_FILENAME)
        if os.path.isfile(module_args_file):
            with open(module_args_file) as f:
                self._NN_MODULE_ARGS = json.load(f)
        self._NN_MODULE_ARGS = {f'module__{k}': v for k, v in self._NN_MODULE_ARGS.items()}
        optimizer_args_file = os.path.join(experiment_folder, self._EXPERIMENT_OPTIMIZER_ARGS_FILENAME)
        if os.path.isfile(optimizer_args_file):
            with open(optimizer_args_file) as f:
                self._NN_OPTIMIZER_ARGS = json.load(f)
        self._NN_OPTIMIZER_ARGS = {f'optimizer__{k}': v for k, v in self._NN_OPTIMIZER_ARGS.items()}
        #TODO: validate loaded json

    def read_create_arguments(self):
        parser = argparse.ArgumentParser(description='Create a dataset')
        parser.add_argument('-i', f'--{self._DICTKEY_INPUT_FILE}', metavar='input file', required=True,
                            type=str, help='path to the input genome')
        parser.add_argument('-a', f'--{self._DICTKEY_ANNOTATIONS}', metavar='annotations file', required=True,
                            type=str, help='path to the TSS annotations for the genome')
        parser.add_argument('-t', f'--{self._DICTKEY_DATASET_TYPE}', metavar='dataset type', required=True,
                            type=str, help='Type of dataset to be created', choices=self._DATASET_CHOICES)
        parser.add_argument('-c', f'--{self._DICTKEY_CONFIGURATION}', metavar='configuration file', required=False,
                            type=str, help='path to the configuration file. If not supplied, defaults will be used')
        parser.add_argument('-e', f'--{self._DICTKEY_EXPERIMENT_FOLDER}', metavar='experiment folder', required=True,
                            type=str, help='path to the experiment folder')

        args = parser.parse_args(sys.argv[3:])
        
        input_file = args.__dict__[self._DICTKEY_INPUT_FILE]
        FastaValidator().validate(input_file)
        self.CONF_DICT[self._DICTKEY_INPUT_FILE] = os.path.relpath(input_file)

        annotations_file = args.__dict__[self._DICTKEY_ANNOTATIONS]
        AnnotationValidator().validate(annotations_file)
        self.CONF_DICT[self._DICTKEY_ANNOTATIONS] = os.path.relpath(annotations_file)

        self.validate_config(args.__dict__[self._DICTKEY_DATASET_TYPE], args.__dict__[self._DICTKEY_CONFIGURATION])

        experiment_folder = args.__dict__[self._DICTKEY_EXPERIMENT_FOLDER]
        FolderValidator().validate(experiment_folder)
        self.CONF_DICT[self._DICTKEY_EXPERIMENT_FOLDER] = os.path.abspath(experiment_folder)

    def read_train_arguments(self):
        parser = argparse.ArgumentParser(description='Train an experiment')
        parser.add_argument('-e', f'--{self._DICTKEY_EXPERIMENT_FOLDER}', metavar='experiment folder', required=True,
                            type=str, help='Path to the experiment folder')
        parser.add_argument('-x', f'--{self._DICTKEY_NN_MAX_EPOCHS}', metavar='max epochs', required=False, default=self._maxepochs_default,
                            type=int, help='Maximum number of epochs')
        parser.add_argument('-p', f'--{self._DICTKEY_NN_PATIENCE}', metavar='patience', required=False, default=self._patience_default,
                            type=int, help='Stop training after this many epochs without progress')
        parser.add_argument('-l', f'--{self._DICTKEY_NN_LEARNING_RATE}', metavar='learning rate', required=False, default=self._learningrate_default,
                            type=int, help='Value for the learning rate parameter')
        parser.add_argument('-b', f'--{self._DICTKEY_NN_BATCH_SIZE}', metavar='batch size', required=False, default=self._batchsize_default,
                            type=int, help='Value for the batch size parameter')
        parser.add_argument('-t', f'--{self._DICTKEY_NN_OPTIMIZER}', metavar='optimizer', required=False, default=self._NN_OPTIMIZERS[0],
                            type=str, help='Optimizer for training the neural network', choices=self._NN_OPTIMIZERS)
        parser.add_argument('-f', f'--{self._DICTKEY_NN_OUTPUT_FUNCTION}', metavar='output function', required=False, default=self._NN_OUTPUT_FUNCTIONS[0],
                            type=str, help='Function of the last layer of the neural network', choices=self._NN_OUTPUT_FUNCTIONS)
        parser.add_argument('-m', f'--{self._DICTKEY_NN_MODULE}', metavar='pytorch neural network module', required=False, default='dprom',
                            type=str, help='Pytorch neural network architecture module to train', choices=self._NN_MODULES)
        args = parser.parse_args(sys.argv[2:])
        experiment_folder = args.__dict__[self._DICTKEY_EXPERIMENT_FOLDER]
        FolderValidator().validate(experiment_folder)
        self.CONF_DICT[self._DICTKEY_EXPERIMENT_FOLDER] = experiment_folder
        for k in [self._DICTKEY_NN_MAX_EPOCHS, self._DICTKEY_NN_PATIENCE,
                self._DICTKEY_NN_LEARNING_RATE, self._DICTKEY_NN_BATCH_SIZE]:
            PositiveValidator().validate(args.__dict__[k])
        args.__dict__.pop(self._DICTKEY_EXPERIMENT_FOLDER)
        return args.__dict__

    def read_convert_arguments(self):
        parser = argparse.ArgumentParser(description='File conversion system')
        parser.add_argument('-i', f'--{self._DICTKEY_INPUT_FILE}', metavar='input file', required=True,
                            type=str, help='Path to the input file')
        parser.add_argument('-o', f'--{self._DICTKEY_OUTPUT_FILE}', metavar='output file', required=True,
                            type=str, help='Path to save the converted file')
        parser.add_argument('-t', f'--{self._DICTKEY_INPUT_TYPE}', metavar='input type', required=True, default=self._FILE_TYPES[0],
                            type=str, help='Type of the input file', choices=self._FILE_TYPES)
        parser.add_argument('-e', f'--{self._DICTKEY_OUTPUT_TYPE}', metavar='output type', required=True, default=self._FILE_TYPES[1],
                            type=str, help='Type of the output file', choices=self._FILE_TYPES)
        args = parser.parse_args(sys.argv[2:])
        input_file = args.__dict__[self._DICTKEY_INPUT_FILE]
        if input_file:
            IsFileValidator().validate(input_file)
        output_file = args.__dict__[self._DICTKEY_OUTPUT_FILE]
        if output_file:
            FileExistsValidator().validate(output_file)
        input_type = args.__dict__[self._DICTKEY_INPUT_TYPE]
        output_type = args.__dict__[self._DICTKEY_OUTPUT_TYPE]
        if input_type == output_type:
            raise RuntimeError(msg="No conversion necessary. Input type = Output type")
        self.CONF_DICT = args.__dict__

    def read_epd_download_arguments(self):
        parser = argparse.ArgumentParser(description='Download annotations from Eukaryotic Promoter Database (EPDnew)')
        parser.add_argument('-o', f'--{self._DICTKEY_OUTPUT_FILE}', metavar='output file', required=False,
                            type=str, help='Path where the downloaded file will be saved')
        parser.add_argument('-d', f'--{self._DICTKEY_EPD_DATABASE}', metavar='database', required=True, default=self._EPD_DATABASES[0],
                            type=str, help='Database (species) to query', choices=self._EPD_DATABASES)
        parser.add_argument('-t', f'--{self._DICTKEY_EPD_TATA_FILTER}', metavar='TATA motif filter', required=True, default=self._EPD_TATA_FILTERS[0],
                            type=str, help='Filter promoters by TATA motif', choices=self._EPD_TATA_FILTERS)
        args = parser.parse_args(sys.argv[3:])
        output_file = args.__dict__[self._DICTKEY_OUTPUT_FILE]
        if output_file:
            FolderValidator().validate(os.path.dirname(output_file))
        return args
    
    def read_f5_download_arguments(self):
        parser = argparse.ArgumentParser(description='Download annotations from Riken Fantom (5) project')
        parser.add_argument('-o', f'--{self._DICTKEY_OUTPUT_FILE}', metavar='output file', required=False,
                            type=str, help='Path where the downloaded file will be saved')
        parser.add_argument('-s', f'--{self._DICTKEY_F5_SPECIES}', metavar='species', required=True, default=self._F5_SPECIES[0],
                            type=str, help='Database (species) to query', choices=self._F5_SPECIES)
        args = parser.parse_args(sys.argv[3:])
        output_file = args.__dict__[self._DICTKEY_OUTPUT_FILE]
        if output_file:
            FolderValidator().validate(os.path.dirname(output_file))
        return args

    def validate_config(self, db_type, config):
        try:
            self.CONF_DICT[self._DICTKEY_DATASET_TYPE] = db_type
            with open(config) as f:
                data = json.load(f)
                configuration = data[self._DICTKEY_CONFIGURATION]
        except TypeError:
            print('No configuration file supplied. Default configuration will be used')
            self.insert_default_values(db_type)
            self.validate_seed()
            return
        except:
            ConfigValidator().validate(config)

        print('Checking dataset configuration:')
        for key in self.CONF_DICT[self._DICTKEY_CONFIGURATION].keys():
            try:
                PositiveValidator().validate(configuration[key])
            except (KeyError, UnboundLocalError):
                raise Exception('Incomplete configuration file detected')
            except ValidationError:
                if key == self._DICTKEY_SEED:
                    continue

        self.CONF_DICT[self._DICTKEY_CONFIGURATION] = configuration
        self.validate_seed()
        if(db_type == self._DATASET_CHOICES[0]): # IO
            self.validate_io(config)
        elif(db_type == self._DATASET_CHOICES[1]): # BIO
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[2]): # BME
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[3]): # CBG
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[4]): # CBPS
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[5]): # LITERATURE
            #TODO
            raise Exception("Functionality currently unavailable")

    def insert_default_values(self, db_type):
        if(db_type == self._DATASET_CHOICES[0]): # IO
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] = self._error_type_default
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN] = self._error_margin_default
        elif(db_type == self._DATASET_CHOICES[1]): # BIO
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[2]): # BME
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[3]): # CBG
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[4]): # CBPS
            #TODO
            raise Exception("Functionality currently unavailable")
        elif(db_type == self._DATASET_CHOICES[5]): # LITERATURE
            #TODO
            raise Exception("Functionality currently unavailable")

    def validate_io(self, config):
        try:
            print('Checking error type in configuration file:')
            if(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[0]): # sequence overlap
                if(len(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN]) != 2):
                    raise Exception('Configuration file error: Error margin is not valid')
            elif(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[1]): # tss proximity
                if(len(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN]) != 1):
                    raise Exception('Configuration file error: Error margin is not valid')
            elif(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[2]): # seq proximity
                if(len(self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN]) != 1):
                    raise Exception('Configuration file error: Error margin is not valid')
            else:
                raise Exception('Configuration file error: Error type is missing or invalid')
                
            print('Checking error margins in configuration file:')
            for value in self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN]:
                PositiveValidator().validate(value)
        except KeyError:
            raise Exception('Incomplete configuration file detected')
        print('Configuration file check completed')

    def validate_seed(self):
        try:
            seed = self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_SEED]
            SeedValidator().validate(seed)
        except KeyError:
            print('Seed not provided in configuration file.')
            seed = 0
        self._set_seeds(seed)

    def save_experiment_config(self):
        save_file = os.path.join(self.CONF_DICT[self._DICTKEY_EXPERIMENT_FOLDER], self._EXPERIMENT_CONFIGURATION_FILENAME)
        self.save_file(save_file, Helper.save_json, json_obj=self.CONF_DICT)

    def ask_and_save_config_parameters(self):
        self.ask_dataset_config()
        self.ask_dataset_type()
        self.ask_error_type()
        save_file = os.path.join(os.getcwd(), 'configuration.json')
        self.save_file(save_file, Helper.save_json, json_obj=self.CONF_DICT)

    def ask_experiment_parameters(self):
        self.ask_config()
        self.ask_dataset_config()
        self.ask_dataset_type()
        self.ask_error_type()

    def ask_convert_parameters(self):
        input_questions = [
            {
                'type': 'input',
                'name': self._DICTKEY_INPUT_FILE,
                'message': 'Input file:',
                'filter': lambda val: os.path.relpath(val),
                'validate': IsFileValidator
            },
            {
                'type': 'list',
                'name': self._DICTKEY_INPUT_TYPE,
                'message': 'Input file type:',
                'choices': self._FILE_TYPES
            }
        ]
        for k, v in prompt(input_questions).items():
            self.CONF_DICT[k] = v
        out_file_types = self._FILE_TYPES.copy()
        out_file_types.remove(self.CONF_DICT[self._DICTKEY_INPUT_TYPE])
        output_questions = [
            {
                'type': 'input',
                'name': self._DICTKEY_OUTPUT_FILE,
                'message': 'Output file:',
                'filter': lambda val: os.path.relpath(val),
                'validate': FileExistsValidator
            },
            {
                'type': 'list',
                'name': self._DICTKEY_OUTPUT_TYPE,
                'message': 'Output file type:',
                'choices': out_file_types # Has the input type removed
            }
        ]
        for k, v in prompt(output_questions).items():
            self.CONF_DICT[k] = v
    
    def ask_config(self):
        questions = [
            {
                'type': 'input',
                'name': self._DICTKEY_INPUT_FILE,
                'message': 'Input file:',
                'filter': lambda val: os.path.relpath(val),
                'validate': FastaValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_ANNOTATIONS,
                'message': 'Annotations file:',
                'filter': lambda val: os.path.relpath(val),
                'validate': AnnotationValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_EXPERIMENT_FOLDER,
                'message': 'Experiment folder:',
                'default': os.path.dirname(os.path.abspath(sys.argv[0])),
                'filter': lambda val: os.path.abspath(val),
                'validate': FolderValidator
            }
        ]

        for k, v in prompt(questions).items():
            self.CONF_DICT[k] = v

    def ask_train_parameters(self):
        questions = [
            {
                'type': 'input',
                'name': self._DICTKEY_EXPERIMENT_FOLDER,
                'message': 'Experiment folder:',
                'default': os.path.dirname(os.path.abspath(sys.argv[0])),
                'filter': lambda val: os.path.abspath(val),
                'validate': FolderValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_OUTPUT_FUNCTION,
                'message': 'Neural network output function:',
                'default': self._NN_OUTPUT_FUNCTIONS[0],
                'validate': FunctionValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_MODULE,
                'message': 'Pytorch module (Neural network architecture):',
                'default': "dprom",
                'validate': ModuleValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_OPTIMIZER,
                'message': 'Pytorch optimizer for training:',
                'default': self._NN_OPTIMIZERS[0],
                'validate': OptimizerValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_MAX_EPOCHS,
                'message': 'Maximum number of epochs during training:',
                'default': self._maxepochs_default,
                'validate': PositiveValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_PATIENCE,
                'message': '(Patience)Stop after this many epochs without progress:',
                'default': self._patience_default,
                'validate': PositiveValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_LEARNING_RATE,
                'message': 'Learning rate:',
                'default': self._learningrate_default,
                'validate': PositiveValidator
            },
            {
                'type': 'input',
                'name': self._DICTKEY_NN_BATCH_SIZE,
                'message': 'Batch size:',
                'default': self._batchsize_default,
                'validate': PositiveValidator
            }
        ]
        train_args = {}
        self.CONF_DICT[self._DICTKEY_EXPERIMENT_FOLDER] = prompt(questions)[self._DICTKEY_EXPERIMENT_FOLDER]
        for k, v in prompt(questions).items():
            if k == self._DICTKEY_EXPERIMENT_FOLDER:
                continue
            train_args[k] = v
        return train_args

    def ask_dataset_config(self):
        questions = [
            {
                'type': 'input',
                'name': self._DICTKEY_SEQ_UPSTREAM_LEN,
                'message': 'Promoter upstream length:',
                'default': str(self._upstream_default),
                'validate': PositiveValidator,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': self._DICTKEY_SEQ_DOWNSTREAM_LEN,
                'message': 'Promoter downstream length:',
                'default': str(self._downstream_default),
                'validate': PositiveValidator,
                'filter': lambda val: int(val)
            },
            {
                'type': 'input',
                'name': self._DICTKEY_STRIDE,
                'message': 'Stride:',
                'default': str(self._stride_default),
                'validate': PositiveValidator,
                'filter': lambda val: int(val)
            }
        ]
        self.CONF_DICT[self._DICTKEY_CONFIGURATION] = {}

        for k, v in prompt(questions).items():
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][k] = v
        self.ask_seed()

    def ask_seed(self):
        question = [
            {
                'type': 'input',
                'name': self._DICTKEY_SEED,
                'message': 'Seed:',
                'default': '0',
                'validate': SeedValidator,
                'filter': lambda val: int(val)
            }
        ]
        self._set_seeds(prompt(question)[self._DICTKEY_SEED])

    def ask_dataset_type(self):
        questions = [
            {
                'type': 'list',
                'name': self._DICTKEY_DATASET_TYPE,
                'message': 'Choose the dataset type:',
                'choices': self._DATASET_CHOICES
            }
        ]

        for k, v in prompt(questions).items():
            self.CONF_DICT[k] = v

    def ask_error_type(self):
        if self.CONF_DICT[self._DICTKEY_DATASET_TYPE] == self._DATASET_CHOICES[0]:
            self.ask_io_error_type_config()
        elif self.CONF_DICT[self._DICTKEY_DATASET_TYPE] == self._DATASET_CHOICES[1]:
            #TODO
            self.ask_bio_error_type_config()

    def ask_io_error_type_config(self):
        questions = [
            {
                'type': 'list',
                'name': self._DICTKEY_ERROR_TYPE,
                'message': 'Select error type:',
                'choices': self._IO_ERROR_TYPES
            }
        ]
        answers = prompt(questions)

        for k, v in answers.items():
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][k] = v

        if self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[0]: # overlap
            questions = [
                {
                    'type': 'input',
                    'name': 'upstream',
                    'message': 'Length of overlap from the 5\' end of promoter:',
                    'validate': PositiveValidator,
                    'filter': lambda val: int(val)
                },
                {
                    'type': 'input',
                    'name': 'downstream',
                    'message': 'Length of overlap from the 3\' end of promoter:',
                    'validate': PositiveValidator,
                    'filter': lambda val: int(val)
                }
            ]
            answers = prompt(questions)
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN] = [answers['upstream'], answers['downstream']]

        elif self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[1]: # tss proximity
            questions = [
                {
                    'type': 'input',
                    'name': 'proximity',
                    'message': 'Length of proximity to the TSS in promoter:',
                    'validate': PositiveValidator,
                    'filter': lambda val: int(val)
                }
            ]
            answers = prompt(questions)
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN] = [answers['proximity']]
        
        elif self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_TYPE] == self._IO_ERROR_TYPES[2]: # tss proximity
            questions = [
                {
                    'type': 'input',
                    'name': 'proximity',
                    'message': 'Length of sequence proximity to the TSS:',
                    'validate': PositiveValidator,
                    'filter': lambda val: int(val)
                }
            ]
            answers = prompt(questions)
            self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_ERROR_MARGIN] = [answers['proximity']]


    def save_file(self, file, save_func, continuous=False, **kwargs):
        if continuous: # file must be buffer
            save_func(file, kwargs)
            return 0

        questions = [
            {
                'type': 'expand',
                'message': 'Conflict on `%s`: ' % file,
                'name': 'overwrite',
                'default': 'y',
                'choices': [
                    {
                        'key': 'y',
                        'name': 'Overwrite',
                        'value': 'overwrite'
                    },
                    {
                        'key': 'a',
                        'name': 'Overwrite this one and all next',
                        'value': 'overwrite_all'
                    },
                    {
                        'key': 'r',
                        'name': 'Rename the file',
                        'value': 'rename'
                    },
                    Separator(),
                    {
                        'key': 'x',
                        'name': 'Abort',
                        'value': 'abort'
                    }
                ]
            }
        ]
        
        if self.file_exists(file):
            if not self._overwrite:
                answers = prompt(questions)
            if answers['overwrite'] == 'overwrite_all':
                self._overwrite = True
            if answers['overwrite'] == 'overwrite' or self._overwrite:
                save_func(file, kwargs)
            elif answers['overwrite'] == 'rename':
                questions = [
                    {
                        'type': 'input',
                        'name': 'file',
                        'message': 'New file name:',
                        'validate': FileExistsValidator
                    }
                ]
                answers = prompt(questions)
                folder = os.path.dirname(os.path.abspath(file))
                save_func(os.path.join(folder, answers['file']), kwargs)
        else:
            save_func(file, kwargs)
        
    def _set_seeds(self, seed):
        if seed == 0:
            print("Seed configuration was set to 0 (This creates a random seed)")
            seed = random.randint(0, 2**32 - 1)
            print("Generating random seed: %d" % seed)
        self.CONF_DICT[self._DICTKEY_CONFIGURATION][self._DICTKEY_SEED] = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_modules(self):
        file_path = os.path.realpath(__file__)
        folder_path = os.path.dirname(file_path)
        modules_folder = os.path.join(folder_path, "modules")
        files = os.listdir(modules_folder)
        module_dict = {}
        for f in files:
            if f[-3:] == ".py" and not f.startswith("__"):
                f_name = f[:-3]
                f_path = os.path.join(modules_folder, f)
                spec = importlib.util.spec_from_file_location(f_name, f_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for x in dir(module):
                    cls = getattr(module, x)
                    if inspect.isclass(cls):
                        module_dict[f_name] = cls
        return module_dict

    @staticmethod
    def save_dataframe(file, kwargs):
        kwargs.get('dataframe').to_csv(file, sep='\t', index=False, header=False)

    @staticmethod
    def save_sld_line(buffer, kwargs):
        line = f"{kwargs.get('start')}\t{kwargs.get('end')}\t+{kwargs.get('pos_strand_label')}\t-{kwargs.get('neg_strand_label')}\n"
        buffer.write(line)
    
    @staticmethod
    def save_fasta_record(buffer, kwargs):
        buffer.write(f">{kwargs.get('record')}{kwargs.get('start')}:{kwargs.get('end')}={kwargs.get('lbl')}\n")
        record_data = kwargs.get('record_data')
        while(len(record_data) > 50):
            line = record_data[:50]
            record_data = record_data[50:]
            buffer.write(f"{line}\n")
        if len(record_data) > 1:
            buffer.write(f"{record_data}\n")
    
    @staticmethod
    def save_sldi_line(buffer, kwargs):
        line = f"{kwargs.get('record')}\t{kwargs.get('record_start')}\t{kwargs.get('record_end')}\n"
        buffer.write(line)
    
    @staticmethod
    def save_aliases_table(buffer, kwargs):
        for row in kwargs.get('aliases_table'):
            alias = row[0]
            chromosome = row[1]
            line = f"{alias}\t{chromosome}\n"
            buffer.write(line)

    @staticmethod
    def save_json(file, json_obj):
        with open(file, 'w') as json_file:
            json.dump(next(iter(json_obj.values())), json_file)

    @staticmethod
    def save_annotation(file, kwargs):
        resp = requests.get(kwargs.get('url'), stream=True)
        with tqdm.wrapattr(open(file, "wb"), "write", miniters=1,
                        total=int(resp.headers.get('content-length', 0)),
                        desc=file) as fout:
            for chunk in resp.iter_content(chunk_size=1024):
                fout.write(chunk)

    @staticmethod
    def save_header(buffer, kwargs):
        line = f"@{kwargs.get('filename')}\n"
        buffer.write(line)

    @staticmethod
    def save_record(buffer, kwargs):
        line = f">{kwargs.get('record')}\n"
        buffer.write(line)

    @staticmethod
    def file_exists(file):
        file_path = Path(file)
        return file_path.exists()

    conversion_table = str.maketrans("ACTGactgN", "TGACtgacN")
    def reverse_complement(self, seq):
        return seq.translate(self.conversion_table)[::-1]

    @staticmethod
    def get_number_of_lines(file):
        f = open(file)                  
        lines = 0
        buf_size = 1048576
        read_f = f.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)
        return lines

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.DNA_DICT), len(seq)), dtype=np.float32)
        for i in range(0, len(seq), self._k):
            token = seq[i:i+self._k]
            one_hot[self.DNA_DICT[token], i] = 1
        return one_hot

    def download_epd_promoters(self, db='human', tata='all', save_path=''): #tata: all, with, without
        headers = {'accept': 'text/html',
                    'content-type': 'application/x-www-form-urlencoded',
                    'sec-ch-ua': '\" Not;A Brand\";v=\"99\", \"Chromium\";v=\"91\"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'same-origin',
                    'sec-fetch-user': '?1',
                    'upgrade-insecure-requests': '1'}
        payload = {'select_db': db, 'idtype': 'epd', 'ids': '', 'tata': tata, 'inrAND': 'AND', 'inr': 'all', 'ccaatAND': 'AND', 'ccaat': 'all', 'gcAND': 'AND', 'gc': 'all', 'dispersionAND': 'AND', 'dispersion': 'all', 'eAverageAND': 'AND', 'eAverage': '', 'eSamplesAND': 'AND', 'eSamples': '', 'action': 'Select', 'database': 'epdnew'}
        base_url = "https://epd.epfl.ch"
        r = requests.post(base_url + "/get_promoters.php", data=payload, headers=headers)
        download_url = ""
        for line in r.text.split('\n'):
            if 'SGA file' in line:
                start = line.index('href=') + 6
                end = line.index('.sga') + 4
                download_url = base_url + line[start:end]
                break
        if not download_url.strip():
            raise Exception("Could not find download link")
        motifs_str = ""
        if tata == self._EPD_TATA_FILTERS[0]:
            motifs_str = "All motifs"
        elif tata == self._EPD_TATA_FILTERS[1]:
            motifs_str = "With TATA motifs (TATA)"
        else:
            motifs_str = "Without TATA motifs (non-TATA)"
        if not save_path:
            save_file = os.path.join(os.getcwd(), f"epd_{db}_tata-{tata}.sga")
        else:
            save_file = save_path
        print(f"Downloading EPD {db} promoters: {motifs_str}")
        self.save_file(save_file, Helper.save_annotation, url=download_url)

    def download_f5_promoters(self, db='human', save_path=''):
        base_url = "https://fantom.gsc.riken.jp/5"
        specie_genome_dict = {'human': 'hg38', 'mouse': 'mm10', 'rat': 'rn6', 'dog': 'canFam3',
                                'chicken': 'galGal5', 'rhesus': 'rheMac8'}
        download_url = ""
        if db in ['human', 'mouse']:
            download_url = base_url + f"/datafiles/reprocessed/{specie_genome_dict[db]}_latest/extra/CAGE_peaks/{specie_genome_dict[db]}_fair+new_CAGE_peaks_phase1and2.bed.gz"
        else:
            download_url = base_url + f"/datafiles/latest/extra/CAGE_peaks/{specie_genome_dict[db]}.cage_peak_coord.bed.gz"
        if not download_url.strip():
            raise Exception("Could not find download link")
        if not save_path:
            save_file = os.path.join(os.getcwd(), f"f5_{db}.bed.gz")
        else:
            save_file = os.path.abspath(save_path)
        print(f"Downloading FANTOM5 {db} promoters")
        self.save_file(save_file, Helper.save_annotation, url=download_url)