import os
import sys
import argparse
import configparser
import logging

import torch
import telegram
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

import data, utils, model


log = logging.getLogger('run_bot')

CONFIG_FILE_PATH = '.config/telegram_conf.ini'


def start(update, context):
    update.message.reply_text('Hi!')

def echo(update, context):
    update.message.reply_text(update.message.text)

def error(update, context):
    log.warning(f"Error: {context.error} in Update {update}")


if __name__ == '__main__':
    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', required=True,
        help="path to pretrained model to load"
    )
    command_args = parser.parse_args()
    # load config file
    if not os.path.exists(CONFIG_FILE_PATH):
        raise FileNotFoundError("Config file not found.")
    config_ini = configparser.ConfigParser()
    config_ini.read(CONFIG_FILE_PATH, encoding='utf-8')


    emb_dict = data.load_emb_dict(os.path.dirname(command_args.model))
    rev_emb_dict = {value: key for key, value in emb_dict.items()}
    end_token = emb_dict[data.END]

    # load pretrained model
    net = model.Seq2seqModel(
        embedding_dim=model.EMBEDDING_DIM,
        hidden_size=model.HIDDEN_SIZE,
        dict_size=len(emb_dict),
    )
    net.load_state_dict(torch.load(command_args.model))

    updater = Updater(config_ini['TELEGRAM']['APIToken'])
    dp = updater.dispatcher

    # add bot handlers
    dp.add_handler(CommandHandler('start', start))
    dp.add_error_handler(error)
    # callback
    def bot_func(update, context):
        tokens = utils.tokenize(update.message.text)
        seq = data.encode_words(tokens, emb_dict)
        message = data.input_mat_to_packed(seq, net.embedding)
        hidden = net.encode(message)
        logits, actions = net.decode_sequences(
            hidden=hidden,
            length=data.MAX_LENGTH,
            init_emb=message.data[0:1],
            mode='sampling',
            end_of_decoding=end_token,
        )
        # omit last end token
        if actions[-1] == end_token:
            actions.pop()
        responce = data.decode_words(actions, rev_emb_dict)
        if responce:
            responce = utils.untokenize(responce)
            update.message.reply_text(responce)

    dp.add_handler(MessageHandler(Filters.text, bot_func))
    log.info("Bot is running")
    updater.start_polling()
    updater.idle()
