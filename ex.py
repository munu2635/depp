import tensorflow as tf
#설정 부분 
x = [0, 3, 7, 5, 1]
min_index = tf.argmin(x)
max_index = tf.argmax(x)
# 실행 부분 
sess = tf.Session()
max_res = sess.run(max_index)
min_res = sess.run(min_index)

print('max_index = ', max_res)
print('min_index = ', min_res)