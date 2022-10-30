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

# Zygote.jl allows us to calculate derivvatives easily
using Zygote
y, back = Zygote.pullback(sin, π);
y
back(1)
gradient(sin,π) == back(1)
