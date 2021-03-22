# ChatBotWithSCST

## What is it?
This is a simple chat bot for entertainment.

The bot learned from Cornell Movie-Dialogue Corpus dataset, using seq2seq model.

The seq2seq model is fine-tuned by REINFORCE algorithm (and self-critical sequence training based on it).

Chat bot is implemented using [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

## Requirements

## Environment
This bot can be both trained and used on Google Colab through notebooks.

## Configurations
- Make directory `./.config`
- Make file `./.config/telegram_conf.ini`

Example format of `telegram_conf.ini`:

```
[TELEGRAM]
APIToken = HERE_IS_YOUR_API_TOKEN [required]
[BOT]
CallName = CALL_NAME_OF_BOT
BotName = REAL_NAME_OF_BOT
...
```
