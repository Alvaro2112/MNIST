
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
import random
import tkinter.simpledialog as tksipledialog
from tkinter import Checkbutton, Button
#from tkinter.simpledialog import askinteger

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

app = tk.Tk()
app.title("AI Number predictor")
app.geometry('400x100')


SAVE_THIS_MODEL = tk.BooleanVar() 
TEST_SET = tk.BooleanVar() 


Checkbutton(app, text='Save Model', var=SAVE_THIS_MODEL).pack()
Checkbutton(app, text='Compute accuracy on test set', var=TEST_SET).pack() 

Button(app, text="Ok",  command=app.destroy).pack()


app.mainloop()

if SAVE_THIS_MODEL.get():
    app = tk.Tk() 
    app.withdraw()
    app.geometry('150x100')
    n_epochs = tk.IntVar() 
    n_epochs = tksipledialog.askinteger(app,"How many epochs?", minvalue=1)
    
    app.destroy()
    app.mainloop()

LOAD_SAVED_MODEL = not SAVE_THIS_MODEL.get()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

def predict():
    image_index = random.randint(1,6666)
    plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
   
    pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    print(pred.argmax(),pred.argmax()==y_test[image_index])
    plt.show()


if bool(SAVE_THIS_MODEL.get()):
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=n_epochs,verbose=0)
    print(model.evaluate(x_test, y_test))

    #serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if bool(LOAD_SAVED_MODEL):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

if bool(TEST_SET.get()):
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    print(model.evaluate(x_test, y_test))




app = tk.Tk() 
app.title("AI Number predictor")
app.geometry('400x100')
Button(app, text="Predict", command=predict).pack()
Button(app, text="Finish",  command=app.destroy).pack()



app.mainloop()

