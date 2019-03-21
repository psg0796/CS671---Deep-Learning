
import numpy as np
import tensorflow as tf

m = np.load("masses.npy")
v = np.load("velocities.npy")
p = np.load("positions.npy")

mass = tf.constant(m, dtype=tf.float32)
pos = tf.placeholder(tf.float32, shape=(100, 2))
vel = tf.placeholder(tf.float32, shape=(100, 2))

time_step = tf.constant(1e-4, dtype=tf.float32)
gravitational_constant = tf.constant(6.67e5, dtype=tf.float32)
threshold = tf.constant(1e-1, dtype=tf.float32)

cond_true = tf.constant(1.0, shape=(100,100), dtype=tf.float32)
cond_false = tf.constant(0.0, shape=(100,100), dtype=tf.float32)
comp = tf.constant(1.0, shape=(1,1), dtype=tf.float32)

diag = np.ones(100,dtype=np.float32)

a = tf.math.square(pos)
b = tf.reduce_sum(a, 1)
c = tf.expand_dims(b, 1)
d = tf.ones(shape=(1, 100))

p1 = tf.matmul(c,d)

e = tf.square(pos)
f = tf.reduce_sum(e , 1)
g = tf.reshape(f, shape=[-1, 1])
h = tf.ones(shape=(100, 1))

p2 = tf.transpose(tf.matmul(g, h, transpose_b=True))

distPair2 = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(pos, pos, transpose_b=True))
diagOne = tf.diag(diag)
distPair = distPair2 + diagOne

posX = tf.slice(pos, [0,0],[100,1])
posY = tf.slice(pos, [0,1],[100,1])

pairWiseXpos = tf.matmul(posX, tf.ones(shape=(1,100))) - tf.transpose(tf.matmul(posX, tf.ones(shape=(100, 1)), transpose_b=True))
pairWiseYpos = tf.matmul(posY, tf.ones(shape=(1,100))) - tf.transpose(tf.matmul(posY, tf.ones(shape=(100, 1)), transpose_b=True))

mass_matrix = tf.transpose(tf.matmul(tf.reshape(mass, shape = [-1, 1]), tf.ones(shape = (100, 1)), transpose_b=True))

pairWiseXpos_mass = tf.math.multiply(mass_matrix, pairWiseXpos)
pairWiseYpos_mass = tf.math.multiply(mass_matrix, pairWiseYpos)

XYM = tf.stack( [pairWiseXpos_mass,pairWiseYpos_mass], axis = 2 )
XYGM = tf.math.scalar_mul(gravitational_constant ,XYM)
power = tf.fill([100,100],-3.0)
distCube = tf.pow(distPair, power)
distCube2 = tf.stack([distCube,distCube],axis = -1)

GMVR3 =  tf.multiply(XYGM,distCube2)
acceleration = tf.reduce_sum(GMVR3,1)

initial_flag = tf.where( tf.math.less(distPair, threshold),cond_true, cond_false )
final_flag = tf.math.reduce_sum(initial_flag)


St = tf.math.add(
    pos,
    tf.math.add(
        tf.math.scalar_mul(time_step,vel), 
        tf.math.scalar_mul(0.5*time_step*time_step, acceleration)
    )
)

Vt = vel + tf.math.scalar_mul(time_step, acceleration)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    abc = sess.run(final_flag, feed_dict = {pos: p})
    while i<236:
        [p, v] = sess.run([St, Vt], feed_dict = {pos: p, vel: v})
        abc = sess.run(final_flag, feed_dict = {pos: p})
        i = i+1
    np.save('posi.npy', p, allow_pickle = False)
    np.save('vel.npy', v, allow_pickle = False)
