#!/usr/bin/env python
import sys
import argparse
from pyfiglet import Figlet

try:
    from .helper import Helper
    from .IO.dataset import IODataset
    from .IO.train import IOTrainer
    from .modules.dprom import DPROMModule
except:
    from helper import Helper
    from IO.dataset import IODataset
    from IO.train import IOTrainer
    from modules.dprom import DPROMModule

class SuprRef:
    _helper: Helper
    def __init__(self):
        f = Figlet(font='isometric2', justify='center')
        print(f.renderText('SUPR REF'))
        self._helper = Helper()
        self.read_cli_arguments()

    def read_cli_arguments(self):
        parser = argparse.ArgumentParser(description='SUpervised PRomoter REcognition Framework Version: 1.0')
        parser.add_argument('command', type=str, help='Subcommand to run', choices=["create", "train", "download"])
        args = parser.parse_args(sys.argv[1:2])
        getattr(self, args.command)()
    
    def create(self):
        choices = ["configuration", "dataset"]
        parser = argparse.ArgumentParser(description='Create a file')
        parser.add_argument('type', type=str, help='Type of file to create', choices=choices)
        args = parser.parse_args(sys.argv[2:3])
        if(args.type == choices[0]): #configuration
            self._helper.ask_and_save_config_parameters()
        if(args.type == choices[1]): #dataset
            if len(sys.argv) > 3:
                self._helper.read_create_arguments()
            else:
                self._helper.ask_experiment_parameters()
            self._helper.save_experiment_config()
        
            if self._helper.CONF_DICT[Helper._DICTKEY_DATASET_TYPE] == Helper._DATASET_CHOICES[0]: # IO
                dataset = IODataset(self._helper, create_dataset=True)
            #TODO add other dataset choices
            elif self._helper.CONF_DICT[Helper._DICTKEY_DATASET_TYPE] == Helper._DATASET_CHOICES[1]: # BIO
                raise Exception("Functionality currently unavailable")
                dataset = BIODataset(self._helper)

    def train(self):
        #TODO: Ask module_args, optimizer_args
        if len(sys.argv) > 2:
            args = self._helper.read_train_arguments()
        else:
            args = self._helper.ask_train_parameters()
        self._helper.load_experiment()
        self._helper.CONF_DICT[Helper._DICTKEY_NN_TRAIN_ARGS] = args
        if self._helper.CONF_DICT[Helper._DICTKEY_DATASET_TYPE] == Helper._DATASET_CHOICES[0]: # IO
            trainer = IOTrainer(self._helper)
            trainer.fit()

    def download(self):
        choices = ["epd", "fantom5"]
        parser = argparse.ArgumentParser(description='Download annotation files')
        parser.add_argument('type', type=str, help='Type of annotation to download', choices=choices)
        args = parser.parse_args(sys.argv[2:3])
        if(args.type == choices[0]): #epd
            params = self._helper.read_epd_download_arguments()
            self._helper.download_epd_promoters(db=params.__dict__[Helper._DICTKEY_EPD_DATABASE],
                                                tata=params.__dict__[Helper._DICTKEY_EPD_TATA_FILTER],
                                                save_path=params.__dict__[Helper._DICTKEY_OUTPUT_FILE])
        if(args.type == choices[1]): #fantom5
            params = self._helper.read_f5_download_arguments()
            self._helper.download_f5_promoters(db=params.__dict__[Helper._DICTKEY_F5_SPECIES],
                                                save_path=params.__dict__[Helper._DICTKEY_OUTPUT_FILE])



def main():
    SuprRef()

if __name__ == '__main__':
    main()