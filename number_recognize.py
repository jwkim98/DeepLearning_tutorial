import numpy as np
from keras import datasets
from keras.utils import np_utils

def data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L,W,H = X_train.shape
    X_train = X_train.reshape(-1,W*H)
    X_test = X_test.reshape(-1,W*H)

    X_train = X_train/255.0
    X_test = X_test/255.0

    return (X_train,Y_train), (X_test,Y_test)

import matplotlib.pyplot as plt
from AnnClass import Ann_Model_C as Model

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc=0)

def main():
    Nin = 784
    Nh = 800
    number_of_class =10
    Nout = number_of_class

    model = Model(Nin,Nh,Nout)
    (X_train, Y_train), (X_test, Y_test) = data_func()

    history = model.fit(X_train,Y_train,epochs=900,batch_size=600,validation_split=0.2, verbose=1)
    performance_test = model.evaluate(X_test, Y_test, batch_size =100)

    print('Test Loss and Accuracy ->', performance_test)

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

if __name__ =='__main__':
    main()
    

    
