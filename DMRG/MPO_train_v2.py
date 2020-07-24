# -*- coding: utf-8 -*-
"""
@author: zfgao
"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data

import MPO_inference_v2 as inference
from MPO_hyperparameter_v2 import *

BATCH_SIZE=FLAGS.batch_size
TRAINING_STEPS=FLAGS.global_step
LEARNING_RATE_BASE=FLAGS.LEARNING_RATE_BASE
LEARNING_RATE_DECAY=FLAGS.LEARNING_RATE_DECAY
REGULARIZER_RATE=FLAGS.REGULARIZER_RATE
MOVING_DECAY=0.99
#seed =12345
#tf.set_random_seed(seed)

def mnist(inp):
    x=tf.compat.v1.placeholder(tf.float32,[None,inference.input_node],name='x-input')
    y_=tf.compat.v1.placeholder(tf.float32,[None,inference.output_node],name='y-input')
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)

    y=inference.inference(x)
    global_step=tf.Variable(0,trainable=False)

    # ema = tf.train.ExponentialMovingAverage(MOVING_DECAY, global_step)
    # ema_op = ema.apply(tf.trainable_variables())

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(input=y_,axis=1))
    loss=tf.reduce_mean(input_tensor=ce)
    loss += tf.add_n([tf.nn.l2_loss(var) for var in tf.compat.v1.trainable_variables()]) * REGULARIZER_RATE
    # loss = loss + tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             inp.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    train_steps=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # with tf.control_dependencies([train_steps,ema_op]):
    #     train_op=tf.no_op(name='train')

    correct_prediction=tf.equal(tf.argmax(input=y,axis=1),tf.argmax(input=y_,axis=1))
    accuracy=tf.reduce_mean(input_tensor=tf.cast(correct_prediction,tf.float32))


    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        best_acc = 0
        for i in range(TRAINING_STEPS):
            xs,ys = inp.train.next_batch(BATCH_SIZE)
            _,step,lr = sess.run([train_steps,global_step,learning_rate],feed_dict={x:xs,y_:ys})
            #if i%1000 == 0:
            accuracy_score = sess.run(accuracy, feed_dict={x:inp.test.images,y_:inp.test.labels})
            #print('step={},lr={}'.format(step,lr))
            if accuracy_score < 0.1:
                print("Guessing")
                break
            if best_acc < accuracy_score:
                best_acc = accuracy_score


                #####################################################
                # Outputs weights purely as numerical weights by layer
                var = [v for v in tf.compat.v1.trainable_variables()]
                weights = []
                for v in var:
                    weights.append(sess.run(v))


                #####################################################
                print('Accuracy at step %s: %s' % (i, accuracy_score))


        accuracy_score=sess.run(accuracy,feed_dict={x:inp.test.images,y_:inp.test.labels})
        print("After %s trainning step(s),best accuracy=%g" %(step,best_acc))
        #####################################################################
        # Resets the parameters
        tf.compat.v1.get_variable_scope().reuse_variables()
        #####################################################################
    return weights

def main(argv=None):
    inp=input_data.read_data_sets("./data/",validation_size=0,one_hot=True)
    weights = mnist(inp)
    return weights

if __name__=='__main__':
    tf.compat.v1.app.run()
