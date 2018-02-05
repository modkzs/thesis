# -*- coding: utf-8 -*-
from RL4match import MatchModel
import numpy as np
import tensorflow as tf

fx = np.transpose(np.load('fuck_x.npy'), (1, 0, 2))

m = MatchModel(300, 10, seed=1)

m.save('fucking_model')

x1 = m.gen_seq(fx)
m.session.close()
tf.reset_default_graph()
del m

m = MatchModel(300, 10, seed=2)
m.load('fucking_model')

x2 = m.gen_seq(fx)

print()
