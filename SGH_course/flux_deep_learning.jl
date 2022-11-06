using Flux

### NEURAL NETWORKS
# there is no way to optimise weight θ in the model. We use the cost function J(θ) in the training process

W = rand(5, 8)
b = rand(5)
model = Dense(5,2)
x, y = rand(5), rand(2);
loss(ŷ, y) = sum((ŷ.- y).^2)/ length(y)
loss(model(x), y)
# We can use one of the flux's loss function:
Flux.mse(model(x),y)

# It is not enough to just define the loss function.
# Good machine learning model has to have a small generalization error

# Because neural network tend to overfit a lot, it is important to use approperiate regularization method.
# Regularization allows the model to approximate data other than the training one

using LinearAlgebra
L₁(θ) = sum(abs, θ)
L₂(θ) = sum(abs2, θ)
J(x,y,W) = Flux.mse(model(x),y) + L₁(W)
J(x,y,W)

# bagging - bootstrap aggregating
# random selection with replacement k samples and estimating k models on them. Then we average out the results

# droput
# We create new models by deleting neurons from the hidden layers with probability p in each iteration.
# in flux droput is implemented as a layer:
model = Chain(Dense(28^2, 32, relu), Dropout(0.1), Dense(32, 10), BatchNorm(64, relu), softmax)


# Optimaization
# SGD (stochastic gradient descent)
# SGD with momentum
# SGD with Nesterov momentum
# AdaGrad (Adaptive Gradient Algorithm)
# ADAM (Adaptive Moment Estimation)

# Flux allows us to calculata the derivative of any function:
f(x) = 3x^2 + 2x + 1
df(x) = gradient(f, x)[1]
df(2)
d²f(x) = gradient(df, x)[1]
d²f(2)

# it works with every function
function pow(x, n)
    r = 1
    for i = 1:n
        r *= x
    end
    return r
end

pow(2, 4)
gradient(x -> pow(x, 3), 5)

# Zygote.jl allows us to calculate derivatives easily
using Zygote
y, back = Zygote.pullback(sin, π);
y
back(1)
gradient(sin,π) == back(1)


# Flux.train!(objective(loss function), data, opt function)
# train works only for one epoch

# we can loop or use an operator:
using Base.Iterators: repeated
dataset = repeated((x, y), 200)


# Example
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, params
using Base.Iterators: repeated
using MLDatasets: MNIST
using ImageCore

imgs, labels = MNIST.traindata();
MNIST.convert2image(imgs)[:,:,5]

labels[5]

# Classify MNIST digits with a simple multi-layer-perceptron

# Stack images into one large batch
X = reshape(float.(imgs),size(imgs,1) * size(imgs,2),size(imgs,3))

# One-hot-encode the labels
Y = onehotbatch(labels, 0:9)


# defining the model
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) 

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

accuracy(X, Y)
# training
Flux.train!(loss, params(m), dataset, opt, cb=throttle(evalcb, 10))

# the best way to store results is in bson
accuracy(X, Y)



m = Chain(
  Dense(28^2, 100, relu),
  Dense(100, 32, relu),
  Dense(32, 10),
  softmax) 

Flux.train!(loss, params(m), dataset, opt, cb=throttle(evalcb, 10))
accuracy(X, Y)



MNIST.convert2image(X)[:,:,2543]
labels[2543]

onecold(m(X), 0:9)[2543]
onecold(Y, 0:9)[2543]