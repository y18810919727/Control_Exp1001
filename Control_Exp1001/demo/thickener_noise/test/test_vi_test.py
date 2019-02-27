#!/usr/bin/python
# -*- coding:utf8 -*-
from unittest import TestCase

import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

from Control_Exp1001.demo.thickener.run import vi_test

class TestVi_test(TestCase):
    def test_vi_test(self):
        vi_test()
