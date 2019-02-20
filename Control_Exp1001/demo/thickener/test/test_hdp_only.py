#!/usr/bin/python
# -*- coding:utf8 -*-
from unittest import TestCase

import numpy as np
import math
import Control_Exp1001 as CE
import os
import json
from Control_Exp1001.demo.thickener.run import hdp_only


class TestHdp_only(TestCase):
    def test_hdp_only(self):
        hdp_only()

