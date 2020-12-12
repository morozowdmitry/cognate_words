#!/usr/bin/env python3

import telebot
from telebot import types

from telegram_utils import db
from telegram_utils import config

import json

import random

import logging

import pandas as pd

from collections import defaultdict


def prepare_new_data(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
    unchecked_pairs = [[word, root, "unchecked", "unchecked"] for root in data_dict.keys() for word in data_dict[root]]
    random.shuffle(unchecked_pairs)
    return pd.DataFrame(unchecked_pairs, columns=["word", "root", "status", "nest"])


def save_data(filename, df):
    df.to_csv(filename)


def pick_unchecked_root(df):
    unckeched_df = df[df["status"] == "unchecked"]
    if unckeched_df.empty:
        return -1
    else:
        idx = unckeched_df.index[0]
        df.loc[idx, "status"] = "on_check"
        return idx


def pick_unnested_word(df):
    unnested_df = df[(~df["status"].isin(["unchecked", "on_check", "Не знаю"])) & (df["nest"] == "unchecked")]
    if unnested_df.empty:
        return -1
    else:
        idx = unnested_df.index[0]
        df.loc[idx, "nest"] = "on_check"
        return idx


def format_question_root(df, user_id, on_check):
    idx = pick_unchecked_root(df)
    if idx == -1:
        msg = f'Извини, вопросов этого типа пока нет. Напиши /start и выбери другой вариант.'
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.row('Ок')

        return msg, markup

    on_check[user_id] = idx

    msg = f'В слове {df.loc[idx, "word"]} есть корень {df.loc[idx, "root"]}?'

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.row('Да', 'Нет', 'Не знаю')

    return msg, markup


def format_question_nest(df, nests, alternations, user_id, on_check):
    idx = pick_unnested_word(df)
    if idx == -1:
        msg = f'Извини, вопросов этого типа пока нет. Напиши /start и выбери другой вариант.'
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.row('Ок')

        return msg, markup

    on_check[user_id] = idx

    possible_root_alternations = alternations[df.loc[idx, "root"]]

    answers = dict()

    for root in possible_root_alternations:
        for nest in nests.get(root, []):
            answers[nest["words"][0]] = nest["id"]

    checked_df = df[~df["nest"].isin(["unchecked", "on_check", "Не знаю"])]
    if checked_df.empty:
        answers["Эти слова не подходят"] = 1
    else:
        answers["Эти слова не подходят"] = checked_df["nest"].max() + 1

    msg = f'Какое из этих слов яваляется однокоренным к слову {df.loc[idx, "word"]}?'

    reply_markup = types.InlineKeyboardMarkup()
    for word, idx in answers.items():
        reply_markup.add(types.InlineKeyboardButton(text=word, callback_data=idx))

    return msg, reply_markup


if __name__ == "__main__":
    with open('tg_key.conf', 'r') as f:
        tg_key = f.read().strip()
    bot = telebot.TeleBot(tg_key)

    df = prepare_new_data("data_for_labeling.json")
    with open('alternations.json', 'r') as f:
        alternations = json.load(f)
    roots_on_check = dict()
    nests_on_check = dict()
    nests = defaultdict(list)

    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        response = ["Привет! Введи код, пожалуйста."]
        bot.send_message(message.from_user.id, "\n".join(response))
        db.set_state(message.chat.id, config.States.S_CODE.value)


    @bot.message_handler(commands=['change'],
                         func=lambda message: db.get_current_state(message.chat.id) in [config.States.S_MODE.value,
                                                                                        config.States.S_NEST.value,
                                                                                        config.States.S_ROOT.value])
    def cmd_change(message):
        response = ["Хорошо, какой режим?"]
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.row('Корни', 'Гнёзда')
        bot.send_message(message.from_user.id, "\n".join(response), reply_markup=markup)
        db.set_state(message.chat.id, config.States.S_MODE.value)


    @bot.message_handler(content_types=['text'],
                         func=lambda message: db.get_current_state(message.chat.id) == config.States.S_CODE.value)
    def cmd_code(message):
        code = message.text.lower()
        if code != 'канапе':
            bot.send_message(message.from_user.id, 'Неверный код. Попробуй ещё раз')
        else:
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
            markup.row('Корни', 'Гнёзда')
            bot.send_message(message.from_user.id, "Какой режим разметки?", reply_markup=markup)
            db.set_state(message.chat.id, config.States.S_MODE.value)


    @bot.message_handler(content_types=['text'],
                         func=lambda message: db.get_current_state(message.chat.id) == config.States.S_MODE.value)
    def select_mode(message):
        if message.text == 'Корни':
            msg, markup = format_question_root(df, message.from_user.id, roots_on_check)
            bot.send_message(message.from_user.id, msg, reply_markup=markup)
            db.set_state(message.chat.id, config.States.S_ROOT.value)
        elif message.text == 'Гнёзда':
            msg, markup = format_question_nest(df, nests, alternations, message.chat.id, nests_on_check)
            bot.send_message(message.chat.id, msg, reply_markup=markup)
            db.set_state(message.chat.id, config.States.S_NEST.value)
        else:
            db.set_state(message.chat.id, config.States.S_START.value)
            bot.send_message(message.from_user.id, 'Что-то пошло не так, давай начнём с начала. Напиши /start.')


    @bot.message_handler(content_types=['text'],
                         func=lambda message: db.get_current_state(message.chat.id) == config.States.S_ROOT.value)
    def process_root_check(message):
        if message.from_user.id not in roots_on_check:
            db.set_state(message.chat.id, config.States.S_START.value)
            bot.send_message(message.from_user.id, 'Что-то пошло не так, давай начнём с начала. Напиши /start.')
        else:
            idx = roots_on_check[message.from_user.id]
            answer = message.text
            df.loc[idx, "status"] = answer
            save_data('checked.csv', df)
            msg, markup = format_question_root(df, message.from_user.id, roots_on_check)
            bot.send_message(message.from_user.id, msg, reply_markup=markup)


    @bot.callback_query_handler(func=lambda call: True and db.get_current_state(call.message.chat.id) == config.States.S_NEST.value)
    def process_cognation(call):
        if call.message.chat.id not in nests_on_check:
            db.set_state(call.message.chat.id, config.States.S_START.value)
            bot.send_message(call.message.chat.id, 'Что-то пошло не так, давай начнём с начала. Напиши /start.')
        else:
            idx = nests_on_check[call.message.chat.id]
            answer = int(call.data)
            df.loc[idx, "nest"] = answer
            nested = False
            for nest in nests.get(df.loc[idx, "root"], []):
                if nest["id"] == answer:
                    nest["words"].append(df.loc[idx, "word"])
                    nested = True
                    break
            if not nested:
                nests[df.loc[idx, "root"]].append({"id": answer, "words": [df.loc[idx, "word"]]})
            save_data('checked.csv', df)
            msg, markup = format_question_nest(df, nests, alternations, call.message.chat.id, nests_on_check)
            bot.send_message(call.message.chat.id, msg, reply_markup=markup)

    bot.polling(none_stop=True, interval=0)
