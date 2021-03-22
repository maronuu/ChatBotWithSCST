import argparse
import logging

import trainer


# HYPER PARAMETERS
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3


log = logging.getLogger("run_train_cross_entropy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--genre', required=True,
        help="Genre to use for training."
    )
    parser.add_argument(
        '--cuda', action='store_true', default=False,
        help="use cuda"
    )
    parser.add_argument(
        '-n', '--name', required=True,
        help="Name of the training. This is used for log name." 
    )
    args = parser.parse_args()

    trainer = trainer.TrainerCrossEntropy(
        genre=args.genre, use_cuda=args.cuda,
        name_of_run=args.name, log=log,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=NUM_EPOCHS,
    )

    trainer.run_train()
