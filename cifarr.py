import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

def get_dataset():

	(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

	Ytrain = to_categorical(Ytrain)
	Ytest = to_categorical(Ytest)
	return Xtrain, Ytrain, Xtest, Ytest

#normalize the images
def pre_processing(train, test):

	train_norm = train.astype('float32')
	test_norm = test.astype('float32')

	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0

	return train_norm, test_norm

# define cnn model
def cnn_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	optimi = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=optimi, loss='categorical_crossentropy', metrics=['accuracy'])
	return model



def model_run():

	Xtrain, Ytrain, Xtest, Ytest = get_dataset()

	Xtrain, Xtest = pre_processing(Xtrain, Xtest)

	model = cnn_model()

	model.fit(Xtrain, Ytrain, epochs=100, batch_size=64)
	# save model
	model.save('cifmodel.h5')


model_run()

