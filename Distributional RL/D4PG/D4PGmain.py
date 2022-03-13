#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import argparse
from DistributedEngine import load_engine
from utils import read_config  # .yml reading tool

CONFIG_PATH = 'Config_Hopper.yml'
def main(config_path):
    # parser = argparse.ArgumentParser(description='Run training')
    # parser.add_argument("--config", type=str, help="Path to the config file.")
    # args = vars(parser.parse_args())
    config = read_config(config_path)
    engine = load_engine(config)
    engine.train()

if __name__ == "__main__":
    main(CONFIG_PATH)






