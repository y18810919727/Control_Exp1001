#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


EXP_NAME = "None"

def _init():
    global EXP_NAME
    EXP_NAME = "None"

def get_exp_name():
    return EXP_NAME

def set_exp_name(new_exp_name):
    global EXP_NAME
    EXP_NAME = new_exp_name