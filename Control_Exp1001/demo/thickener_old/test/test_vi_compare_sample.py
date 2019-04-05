#!/usr/bin/python
# -*- coding:utf8 -*-
from unittest import TestCase

import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.run import vi_compare_sample

class TestVi_compare_sample(TestCase):
    def test_vi_compare_sample(self):
        vi_compare_sample()
