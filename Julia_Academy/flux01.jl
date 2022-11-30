using Flux
model = Dense(1,1) # flux automatically sets the random weight value for the layers
model.weight
model.bias

# Chain() function allows us to connect multiple layers together
layer1 = Dense(1,1)
layer2 = Dense(1,1)
model = Chain(layer1, layer2)

# Convolutional layer
layer2 = Conv((5,5), 3 => 7, relu) # filter 5x5, input => output channel sizes, activation function

# data
xs = rand(Float32, 100, 100, 3, 50);
xs[1]
layer2(xs)
