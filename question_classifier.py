import argparse
import training

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

args = parser.parse_args()
if args.train:
    pass
# call train function
# train(args.config)
elif args.test:
    pass
#call test function
# test(args.config)