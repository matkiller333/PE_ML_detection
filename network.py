import tensorflow as tf


def create_model():
	model = tf.keras.Sequential([  # creates the architecture of the network
		tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
		tf.keras.layers.MaxPooling2D((2, 2)),
		tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
		tf.keras.layers.MaxPooling2D((2, 2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(32, activation='relu'),
		tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

	return model


def train_model(model, train_images, train_labels, epoch_num):
	model.fit(train_images, train_labels, epochs=epoch_num)


def test_model(model, test_images, test_labels):
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print('\nTest accuracy:', test_acc)

	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(test_images)


def save_weights(model, weights_path='./checkpoints/weights'):
	model.save_weights(weights_path)


def load_weights(weights_path, model):
	return model.load_weights(weights_path)
