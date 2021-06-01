import os
import argparse
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import GEC


def main(args):
    gec_model = GEC(pretrained_weights_path=args.weights)
    print(gec_model.correct(args.text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        help='Path to model weights',
                        required=True)
    parser.add_argument('-t', '--text',
                        help='Text to correct',
                        required=True)
    args = parser.parse_args()
    main(args)
