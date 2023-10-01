# coding: utf-8
# @Author: Zyl
# @Email: zshigerc@163.com
# @Place: CUG
# @Time: 2023/7/4 15:20

import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn

label_emb = nn.Embedding(10, 10)
print(label_emb(1))
gen_input = jt.contrib.concat((label_emb([1]), jt.array([[0.10] * 10])), dim=1)
print(gen_input)
