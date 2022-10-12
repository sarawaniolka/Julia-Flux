using Flux, Plots
using CSV
using DataFrames
using Pkg
using Flux: params
model = Dense(2,1, σ)
model.weight
model.bias
typeof(model.weight)
x = rand(2)
model(x)
σ.(model.weight*x + model.bias)
methods(Flux.mse)


# data
apples = CSV.read("apples.dat.txt", DataFrame; normalizenames=true)
bananas = CSV.read("banana.dat.txt", DataFrame; normalizenames=true)
x_apples = [ [row.red, row.green] for row in eachrow(apples)]
x_bananas = [ [row.red, row.green] for row in eachrow(bananas)]

# combine data
xs = [x_apples; x_bananas]
ys = [fill(0, size(x_apples)); fill(1,size(x_bananas))]

# model
model = Dense(2, 1, σ)
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
contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[1], fill=true)
scatter!(first.(x_apples), last.(x_apples), label="apples")
scatter!(first.(x_bananas), last.(x_bananas), label="bananas")
xlabel!("mean red value")
ylabel!("mean green value")