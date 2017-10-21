# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import csv

# fix random seed for reproducibility
np.random.seed(49)

def convtofloat(s):
    try:
        return float(s)
    except ValueError:
        return 0.

# load pima indians dataset
dataset = np.loadtxt("tracking_person.csv", delimiter=",",converters={0:convtofloat,1:convtofloat,2:convtofloat,3:convtofloat,4:convtofloat})



# split into input (X) and output (Y) variables
X = dataset[:,0:81]
Y = dataset[:,81]

adam=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
RSM=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
# create model
model = Sequential()
model.add(Dense(3, input_dim=81, activation='softmax'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
# Compile model
#model.add(Dense(1, activation='softmax'))
# Can use SGD as well... Stochastic gradient descent optimizer. In poisson, to feel the power
# model.compile(loss='mse',optimizer=RSM, metrics=['accuracy'])
#model.compile(loss='poisson', optimizer=adam, metrics=['accuracy'])
#model.compile(loss='kullback_leibler_divergence', optimizer=RSM, metrics=['accuracy'])
# model.compile(loss='logcosh', optimizer=RSM, metrics=['accuracy'])
#
# model.fit(X, Y, epochs=15000, batch_size=4)
# scores = model.evaluate(X, Y)


# categorical_crossentropy
model.add(Dense(23, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(Y, num_classes=23)
model.fit(X, categorical_labels, epochs=1000, batch_size=4)
scores = model.evaluate(X, categorical_labels)

# evaluate the model
# serialize model to JSON
model_json = model.to_json()
with open("modelsoftmax_matrix.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelsoftmax_matrix.h5")
print("Saved model to disk")
