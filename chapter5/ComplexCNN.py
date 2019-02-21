#!/usr/bin/env python
#-*-coding:utf-8-*-
import os,sys
sys.path.append(os.pardir)
import tensorflow as tf
import numpy as np
import time
from cifar10 import cifar10,cifar10_input


cifar10.maybe_download_and_extract()

