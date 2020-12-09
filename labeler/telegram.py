#!/usr/bin/env python3

import telebot
from telebot import types

from telegram_utils import db
from telegram_utils import config

import json

import random

import logging


def prepare_new_data(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    unchecked_pairs = [[word, root, "unchecked"] for root in data_dict.keys() for word in data_dict[root]]
    return unchecked_pairs


def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pick_random_unchecked(pairs):
    while True:
        answer_idx = random.randint(0, len(pairs))
        answer = pairs[answer_idx]
        if answer[2] == "unchecked":
            return answer, answer_idx


def format_question(pairs, chat_id, on_check):
    word, idx = pick_random_unchecked(pairs)

    on_check[chat_id] = {'word': word[0],
                         'root': word[1],
                         'idx': idx}

    msg = f'В слове {word[0]} есть корень {word[1]}, верно?'

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.row('Да', 'Нет', 'Не знаю')

    return msg, markup


if __name__ == "__main__":
    with open('tg_key.conf', 'r') as f:
        tg_key = f.read().strip()
    bot = telebot.TeleBot(tg_key)

    unchecked_data = prepare_new_data("data_for_labeling.json")
    checked_data = list()
    on_check = dict()

    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        response = ["Привет! Введи код, пожалуйста."]
        bot.send_message(message.from_user.id, "\n".join(response))
        db.set_state(message.chat.id, config.States.S_CODE.value)


    @bot.message_handler(content_types=['text'],
                         func=lambda message: db.get_current_state(message.chat.id) == config.States.S_CODE.value)
    def cmd_code(message):
        code = message.text.lower()
        if code != 'канапе':
            bot.send_message(message.from_user.id, 'Неверный код. Попробуй ещё раз')
        else:
            msg, markup = format_question(unchecked_data, message.chat.id, on_check)
            bot.send_message(message.from_user.id, msg, reply_markup=markup)
            db.set_state(message.chat.id, config.States.S_ANSWER.value)


    @bot.message_handler(content_types=['text'],
                         func=lambda message: db.get_current_state(message.chat.id) == config.States.S_ANSWER.value)
    def process_answer(message):
        if message.chat.id not in on_check:
            db.set_state(message.chat.id, config.States.S_START.value)
            bot.send_message(message.from_user.id, 'Что-то пошло не так, давай начнём с начала. Напиши /start.')
        else:
            question = on_check[message.chat.id]
            answer = message.text
            question['status'] = answer
            checked_data.append(question)
            save_data('checked.json', checked_data)
            unchecked_data[question["idx"]][2] = "checked"
            msg, markup = format_question(unchecked_data, message.chat.id, on_check)
            bot.send_message(message.from_user.id, msg, reply_markup=markup)

    bot.polling(none_stop=True, interval=0)
