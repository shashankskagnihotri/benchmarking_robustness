import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', required=True, help='Specify a architecture')
# parser.add_argument('--mode', required=True, help='Specify whether to train or test the model', choices=['train', 'test'])

args, unknown = parser.parse_known_args()
name = args.name
print(name)


if args.architecture.lower() == "cfnet":
    
    if args.mode.lower() == "train":
        from CFNet import main
        main.train()
    else:
        from CFNet import test
        main.test()