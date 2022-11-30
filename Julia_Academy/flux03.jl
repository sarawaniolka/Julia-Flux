# 1 neuron with many outputs
# distinguishing between apples, bannanas and grapes
# imports
using CSV, DataFrames
using Flux
using Flux: onehot, params
using Plots

# import data
apples1 = CSV.read("Apple_Golden_1.dat.txt", DataFrame; normalizenames=true);
apples2 = CSV.read("Apple_Golden_2.dat.txt", DataFrame; normalizenames=true);
apples3 = CSV.read("Apple_Golden_3.dat.txt", DataFrame; normalizenames=true);
bananas = CSV.read("banana.dat.txt", DataFrame; normalizenames=true);
grapes1 = CSV.read("Grape_White.dat.txt", DataFrame; normalizenames=true);
grapes2 = CSV.read("Grape_White_2.dat.txt", DataFrame; normalizenames=true);
apples = vcat(apples1, apples2, apples3)
grapes = vcat(grapes1, grapes2)
# rows into vectors
x_apples = [ [row.red, row.green] for row in eachrow(apples)]
x_bananas = [[row.red, row.green] for row in eachrow(bananas)]
x_grapes = [[row.red, row.green] for row in eachrow(grapes)]

# combine data
xs = vcat(x_apples, x_bananas, x_grapes)

# manually
ys = vcat(fill([1,0,0], size(x_apples)), 
        fill([0,1,0], size(x_bananas)), 
        fill([0,0,1], size(x_grapes)))

# one-hot vectors
ys = vcat(fill(onehot(1, 1:3), size(x_apples)),
          fill(onehot(2, 1:3), size(x_bananas)),
          fill(onehot(3, 1:3), size(x_grapes)))


# model
model = Dense(2, 3, σ)
model(xs[end])

L(x,y) = Flux.mse(model(x),y)
# Opt = SGD(params(model)) does not work
Opt = Descent()
data = zip(xs,ys)
mparams = params(model)
        for step in 1:100
	       Flux.train!(L, mparams, data, Opt)
        end


# visualization
plot()
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y])[1], levels=[0.5, 0.51], color = cgrad([:blue, :blue]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y])[2], levels=[0.5,0.51], color = cgrad([:green, :green]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y])[3], levels=[0.5,0.51], color = cgrad([:red, :red]))
scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples", color = :blue)
scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas", color = :green)
scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes", color = :red)