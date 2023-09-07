import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0])
plt.show()

# =====
# print(x_train[0])
# ---
# plt.imshow(x_train[0], cmap='gray')
# plt.imshow(x_train[0], cmap=plt.cm.gray)
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
# ---
print(y_train[0])
# =====


# ====================================================================

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# -----

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

# ====================================================================

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)
print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
