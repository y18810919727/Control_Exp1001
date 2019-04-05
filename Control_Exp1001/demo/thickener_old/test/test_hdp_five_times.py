#!/usr/bin/python
# -*- coding:utf8 -*-
from unittest import TestCase

import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.run import hdp_five_times


class TestHdp_five_times(TestCase):
    def test_hdp_five_times(self):
        hdp_five_times()
