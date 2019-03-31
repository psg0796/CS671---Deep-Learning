import tensorflow as tf
import numpy as np
import sklearn

filepath = 'model6/model'

model = tf.keras.models.load_model(
    filepath,
    compile=True
)
inputPath = 'inputData/'

(x_train, y_train_length, y_train_width, y_train_color, y_train_angle) = (np.load(inputPath + 'x_train.npy'), np.load(inputPath + 'y_train_length.npy'), np.load(inputPath + 'y_train_width.npy'), np.load(inputPath + 'y_train_color.npy'), np.load(inputPath + 'y_train_angle.npy'))
(x_test, y_test_length, y_test_width, y_test_color, y_test_angle) = (np.load(inputPath + 'x_test.npy'), np.load(inputPath + 'y_test_length.npy'), np.load(inputPath + 'y_test_width.npy'), np.load(inputPath + 'y_test_color.npy'), np.load(inputPath + 'y_test_angle.npy'))

res = model.predict(x_test)
labels = [y_test_length, y_test_width, y_test_color, y_test_angle]
predictions = [[],[],[],[]]
cnf_mtx = [[],[],[],[]]
f1_score = [[],[],[],[]]

i = 0
for res0 in res:
	for res1 in res0:
		predictions[i].append(np.argmax(res1))
	i = i + 1
sess = tf.Session()
with sess.as_default():
  for i in range(4):
    labels_each_head = labels[i]
    predictions_each_head = predictions[i]

    cnf_mtx[i] = tf.math.confusion_matrix(
      labels_each_head,
      predictions_each_head
    ).eval()
    
    f1_score[i] = sklearn.metrics.f1_score(
      labels_each_head,
      predictions_each_head,
	  average = 'micro'
    )

np.save('cnf_mtx',np.asarray(cnf_mtx))
np.save('f1_score',np.asarray(f1_score))
