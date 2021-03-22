import argparse
import logging

from . import trainer


# HYPER PARAMETERS
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 5e-4


log = logging.getLogger("run_train_reinforce")


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
    parser.add_argument(
        '--load', required=True,
        help='path to pretrained seq2seq model (state_dict)'
    )
    parser.add_argument(
        '--samples', type=int, default=4,
        help='The number of samples trainer samples in RL.'
    )
    args = parser.parse_args()

    trainer = trainer.TrainerReinforce(
        genre=args.genre, use_cuda=args.cuda,
        name_of_run=args.name, log=log,
        model_path=args.load,
        n_samples=args.samples,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=NUM_EPOCHS,
    )

    trainer.run_train()
