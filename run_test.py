import logging
import argparse

import torch

import data, model, utils

### logging
log = logging.getLogger("run_test")

if __name__=='__main__':
    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--genre', required=True,
        help="Genre to use for test."
    )
    parser.add_argument(
        '-l', '--load', required=True,
        help="Model path to load."
    )
    args = parser.parse_args()

    # load data
    dialogues, emb_dict = data.load_dialogue_and_dict(genre_filter=args.genre)
    test_data = data.encode_phrase_pairs(dialogues, emb_dict)
    