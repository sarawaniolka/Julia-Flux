using DelimitedFiles
using Statistics
using StatsBase
using DataFrames
using Flux
using Plots
using CSV

# read the file
data = DataFrame(CSV.File("titanic.csv"))

# get to know the data
names(data)
describe(data)

grp = groupby(data, "Survived")
combine(grp, nrow, vec(valuecols(grp) .=> [mean]))
countmap(data[!,"Sex"])
proportions(data[!, :Survived])

countmap(data[!, :Survived])

# missing values
data = data[completecases(data), :]
# 2 rows deleted
# test and training

train_ratio = 0.7
train_set = data[1:floor(Int,size(data,1)*train_ratio),:];
test_set = data[floor(Int,size(data,1)*train_ratio + 1):end,:];

show(train_set)

# the data is transposed - we are using algebra
X_train = Matrix(train_set[:,1:end-1])';
X_test = Matrix(test_set[:,1:end-1])';
y_train = train_set[!, :Survived];
y_test = test_set[!, :Survived];

describe(X_test)
X_test
# data normalization - you must normalise the data if you want to train the nn
function scale(X)
    m = mean(X, dims=2) #mu
    s = std(X, dims=2)

    X_norm = (X .- m) ./ s
    return (X_norm, m, s);
end

# my data is already split, so I do it this way:
function scale(X, m, s)
    X_norm = (X .- m) ./ s
    return X_norm;
end
# Julia allows us to create functions with the same names with methods for different types of data

# starting with the train set bc it is bigger
X_train, m, s = scale(X_train);
X_test = scale(X_test, m, s);

# defining the weights of the model and the sigmoidal function
β = rand(1, size(X_train, 1) +1);
print(β)
Predict(β, x) = 1 ./ (1 .+ exp.(-β[1:end-1]' * x .-β[end]))


Predict(β, X_train) # for entire dataset
Predict(β, X_train[:,1]) # for a specific observation

# LEARNING PROCESS
# binary cross-entropy (log-loss) -> most common function for classification problems
# we want to find a function that will perfectly seperate our two categories
L(ŷ, y) = (-y') * log.(ŷ') - (1 .- y') * log.(1 .- ŷ')


function simple_∇(β, X, y)
    J = L(Predict(β, X),y)[1] # cost function (avg of the loss function), we don't need to divide by the length of the vector, it's a linear transformation
    ∇ = Float64[] # gradient
    for i = 1:length(β)
        b = β[i]
        β′ = β .+ (LinearIndices(β) .== i) * b * √eps()
        β′′ = β .- (LinearIndices(β) .== i) * b * √eps()
        Δf = (L(Predict(β′,X),y)[1] - L(Predict(β′′,X),y)[1]) / (2*b*√eps())
        push!(∇,Δf)
    end
    return J, ∇
end

# optimising the weights β
# ƞ - learning rate
# ϵ - stopping criterion
function solve!(β, X, y; ƞ = 0.001, ϵ= 10^-10, maxit = 50_000)
    iter = 1
    Js = Float64[]
    J, ∇ = simple_∇(β, X, y)
    push!(Js, J)
    while true
        β₀ = deepcopy(β) # copying instead of referencing
        β .-= ƞ * ∇'
        J, ∇ = simple_∇(β, X, y)
        push!(Js, J)
        stop = maximum(abs.(β .- β₀))
        stop < ϵ && break
        iter += 1
        iter > maxit  && break
    end
    return Js
end

Js = solve!(β, X_train, y_train);
plot(Js[1:100])
accuracy(β, X, y, T = 0.5) = sum((Predict(β, X)' .≥ T ).== y)/length(y)
accuracy(β, X_test, y_test)