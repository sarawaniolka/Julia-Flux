using Flux, Statistics
using CSV, DataFrames
using Plots
using Flux
using Flux: Data.DataLoader
using Flux: @epochs
using CUDA
using Random
using IterTools: ncycle

# load data
iris = CSV.read("iris.csv", DataFrame; normalizenames=true);
describe(iris)

plotly()

dropmissing!(iris)

df_species = groupby(iris, :variety)


# visualization
scatter(title="length vs width", xlabel = "length", ylab = "width", iris.sepal_length, iris.sepal_width, label = "sepal")
scatter!(iris.petal_length, iris.petal_width, label = "petal", color="blue")


# Sepal for each specie
scatter(title="length vs width", xlabel = "length", ylab = "width", df_species[("Setosa",)].sepal_length, df_species[("Setosa",)].sepal_width, label = "Setosa")
scatter!(df_species[("Versicolor",)].sepal_length, df_species[("Versicolor",)].sepal_width, label = "Versicolor")
scatter!(df_species[("Virginica",)].sepal_length, df_species[("Virginica",)].sepal_width, label = "Virginica")

# Patel for each specie
scatter(title="length vs width", xlabel = "length", ylab = "width", df_species[("Setosa",)].petal_length, df_species[("Setosa",)].petal_width, label = "Setosa")
scatter!(df_species[("Versicolor",)].petal_length, df_species[("Versicolor",)].petal_width, label = "Versicolor")
scatter!(df_species[("Virginica",)].petal_length, df_species[("Virginica",)].petal_width, label = "Virginica")

using Flux: onecold
Random.seed!(123);

# Shuffle
data = convert(Array, iris)
data = data[shuffle(1:end), :]

# train/test
train_ratio = .7
# train_set = data[1:floor(Int,size(data,1)*train_ratio),:];
# test_set = data[floor(Int,size(data,1)*train_ratio + 1):end,:];
idx = Int(floor(size(iris, 1) * train_ratio))
data_train = data[1:idx,:]
data_test = data[idx+1:end, :]

# get features
get_feat(d) = transpose(convert(Array{Float32},d[:,1:end-1]))
x_train = get_feat(data_train)
x_test = get_feat(data_test)

# encoding
onehot(d) = Flux.onehotbatch(d[:,end], unique(iris.variety))
y_train = onehot(data_train)
y_test = onehot(data_test)



# creating batches
batch_size= 1
train_dl = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
test_dl = DataLoader((x_test, y_test), batchsize=batch_size)


loss(x,y) = Flux.Losses.logitbinarycrossentropy(m(x), y)


# the model
# Layers: 4, 8 -> 8,3 
# Loss: logit binary crossentropy
# Optimizer: ADAM
# Learning rate: 0.01

m = Chain(Dense(4,8, relu), Dense(8,3), softmax)

lr = 0.01
opt = ADAM(lr)
params = Flux.params(m)

function loss_all(data_loader)
    sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader) 
end


function acc(data_loader)
    f(x) = Flux.onecold(cpu(x))
    acces = [sum(f(m(x)) .== f(y)) / size(x,2)  for (x,y) in data_loader]
    sum(acces) / length(data_loader)
end

train_losses = []
test_losses = []
train_acces = []
test_acces = []

epochs = 30
callbacks = [
    () -> push!(train_losses, loss_all(train_dl)),
    () -> push!(test_losses, loss_all(test_dl)),
    () -> push!(train_acces, acc(train_dl)),
    () -> push!(test_acces, acc(test_dl)),
]



for epoch in 1:epochs
    Flux.train!(loss, params, train_dl, opt, cb = callbacks)
end

@show train_loss = loss_all(train_dl)
@show test_loss = loss_all(test_dl)
@show train_acc = acc(train_dl)
@show test_acc = acc(test_dl)

y = (y_test[:,1])
pred = (m(x_test[:,1]))

x_axis = 1:epochs * size(y_train,2)
plot(x_axis, train_losses, label = "Training loss", title = "Loss", xaxis = "epochs * data size")

