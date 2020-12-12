# -*- coding: utf-8 -*-

from enum import Enum

token = "1234567:ABCxyz"
db_file = "database.vdb"


class States(Enum):
    """
    Мы используем БД Vedis, в которой хранимые значения всегда строки,
    поэтому и тут будем использовать тоже строки (str)
    """
    S_START = "0"  # Начало нового диалога
    S_CODE = "1"
    S_NEST = "2"
    S_ROOT = "3"
    S_MODE = "4"
