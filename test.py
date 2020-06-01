import tensorflow as tf

one = (2,3)
# print('one: ', one.shape)
test = tf.zeros((2,3) + one) # + one)
print('test: ', test.shape)
