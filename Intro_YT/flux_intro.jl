# packages
using MLDatasets
using Flux, Images, Plots
using Flux: crossentropy, onecold, onehotbatch, train!, params
using LinearAlgebra, Random, Statistics

 # Load the dataset
X_train_raw, y_train_raw = MNIST(:train)[:]
X_test_raw, y_test_raw = MNIST(:test)[:]

X_train_raw

# view the Images
index = 1
img = X_train_raw[:,:, index]
colorview(Gray, img')

# view training label
y_train_raw
y_train_raw[index]
X_test_raw
img = X_test_raw[:,:,index]
colorview(Gray, img')
y_test_raw[index]


# data pre-processing
#converting the image into a vector
X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

# one-hot encoding
y_train = onehotbatch(y_train_raw, 0:9)
y_test = onehotbatch(y_test_raw, 0:9)


# define model architecture
# input: 784, hidden: 32, output: 10
model = Chain(
    Dense( 28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)

# define loss function
loss(x, y) = crossentropy(model(x), y)

# track parameters
ps = params(model)

# select optimizer
learning_rate = 0.01
opt = ADAM(learning_rate)

# train model
loss_history = []
epochs = 500

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    # print reports
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end

# make predi train_loss
y_hat_raw = model(X_test)

# convert matrix to column vector
y_hat = onecold(y_hat_raw).-1
y = y_test_raw
# accuracy
mean(y_hat.==y)

# display results
check = [y_hat[i] == y[i] for i in 1:length(y)]
index = collect(1:length(y))
check_display = [index y_hat y check]
vscodedisplay(check_display)

# view misclassification
misclas_index = 9
img = X_test_raw[:, :, misclas_index]
colorview(Gray, img')
y[misclas_index]
y_hat[misclas_index]

# plot loss
gr(size = (600, 600))

# plot learning curve
p_l_curve = plot(1:epochs, loss_history,
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2
    )
    
