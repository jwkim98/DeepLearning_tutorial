from keras import*

class Ann_Model_A(models.Model):
    def __init__ (self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape = (Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x,y)
        self.compile(loss = 'categorical_crossentropy',
                     optimizer = 'adam', metrics=['accuracy'])

class Ann_Model_B(models.Model):
    def __init__ (self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('tanh')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape = (Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x,y)
        self.compile(loss = 'categorical_crossentropy',
                     optimizer = 'sgd', metrics=['accuracy'])

class Ann_Model_C(models.Sequential):
    def __init__ (self,Nin, Nh, Nout):
        hidden1 = layers.Dense(Nh, activation = 'tanh', input_shape = (Nin,) )
        dropout1 = layers.Dropout(0.5)
        hidden2 = layers.Dense(800, activation = 'tanh')
        dropout2 = layers.Dropout(0.5)
        hidden3 = layers.Dense(Nh, activation = 'tanh')
        dropout3 = layers.Dropout(0.5)
        output = layers.Dense(Nout, activation = 'softmax')

        super().__init__()

        self.add(hidden1)
        self.add(dropout1)
        self.add(hidden2)
        self.add(dropout2)
        self.add(hidden3)
        self.add(dropout3)
        self.add(output)
        
        sgd = optimizers.SGD(lr = 0.01, momentum = 0.9)
        adagrad = optimizers.Adagrad(lr = 0.01)
        
        self.compile(loss = 'categorical_crossentropy', optimizer = sgd , metrics = ['accuracy'])

