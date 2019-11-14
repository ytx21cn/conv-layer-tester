from keras import layers, models

# model source: https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d

def main():
    input_shape = (32, 32, 3)
    MaxPool2D_filter_size = (2, 2)

    model = models.Sequential()

    model.add(layers.Conv2D(8, (5, 5), input_shape=input_shape)) # CONV1 (filter_size=5, stride=1)
    model.add(layers.MaxPool2D(MaxPool2D_filter_size)) # POOL1
    model.add(layers.Conv2D(16, (5, 5))) # CONV2 (filter_size=5, stride=1)
    model.add(layers.MaxPool2D(MaxPool2D_filter_size)) # POOL2

    model.add(layers.Flatten())

    model.add(layers.Dense(120)) # FC3
    model.add(layers.Dense(84)) # FC4
    model.add(layers.Dense(10, activation='softmax')) # Softmax layer

    model.summary()


if __name__ == '__main__':
    main()
