
import numpy as np
import tensorflow as tf
import MPO_Net_FC2.tt_v2 as tt
from MPO_Net_FC2.FC2_hyperparameter_v2 import *

# Required to work in Jupyter
tf.compat.v1.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS(sys.argv, known_only=True)

def inference(inputs, r_1, r_2):
    #r_1 = FLAGS.tt_ranks_1
    #r_2 = FLAGS.tt_ranks_2
    input_node=FLAGS.input_node
    output_node=FLAGS.output_node
    hidden1_node=FLAGS.hidden_node

    #TTO_layer1
    inp_modes1 =  [4,7,7,4]
    out_modes1 =  [4,4,4,4]
    mat_rank1  =  [1,r_1,r_1,r_1,1]
    # inp_modes1 = [2, 2, 7, 7, 2, 2]
    # out_modes1 = [2, 2, 2, 2, 2, 2]
    # mat_rank1 = [1, r_1, r_1, r_1, r_1, r_1, 1]

    #TTO_layer2
    inp_modes2 = [4,4,4,4]
    out_modes2 = [1,10,1,1]
    mat_rank2 =  [1,r_2,r_2,r_2,1]
    # inp_modes2 = [2, 2, 2, 2, 2, 2]
    # out_modes2 = [1, 1, 2, 5, 1, 1]
    # mat_rank2 = [1, r_2, r_2, r_2, r_2, r_2, 1]

    inputs = tt.tto(inputs,
                    np.array(inp_modes1,dtype=np.int32),
                    np.array(out_modes1,dtype=np.int32),
                    np.array(mat_rank1,dtype=np.int32),
                    scope='tt_scope_1')
    inputs = tf.nn.relu(inputs)
    inputs = tt.tto(inputs,
                    np.array(inp_modes2, dtype=np.int32),
                    np.array(out_modes2, dtype=np.int32),
                    np.array(mat_rank2, dtype=np.int32),
                    scope='tt_scope_2')
    return inputs
