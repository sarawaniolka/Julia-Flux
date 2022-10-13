# 1 neuron with many outputs
# distinguishing between apples, bannanas and grapes
# imports
using CSV, DataFrames
using Flux

# import data
apples = CSV.read("apples.dat.txt", DataFrame; normalizenames=true)
bananas = CSV.read("banana.dat.txt", DataFrame; normalizenames=true)
grapes = CSV.read("Grape_White.dat.txt", DataFrame; normalizenames=true)

# rows into vectors
apples
x_apples = [ [row.red, row.green] for row in eachrow(apples)]
x_apples
x_bananas = [[row.red, row.green] for row in eachrow(bananas)]
x_grapes = [[row.red, row.green] for row in eachrow(grapes)]

# combine data
xs = [x_apples, x_bananas, x_grapes]
xs


