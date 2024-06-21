import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', required=True, help='Specify an architecture')
parser.add_argument('-m', '--mode', required=True, help='Specify whether to train or test the model', choices=['train', 'test'])

args, unknown = parser.parse_known_args()
# name = args.name
# print(name)

# Add the CFNet directory to the PYTHONPATH
cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CFNet'))
sys.path.append(cfnet_path)

if args.architecture.lower() == "cfnet":
    
    if args.mode.lower() == "train":
        from CFNet import main
        main.train()
    else:
        from CFNet import test
        main.test()