# Classification task on the Australian credit scoring data. 
# Goal: Distinguish between the good and bad debtors

using DelimitedFiles
using PyPlot
using Statistics
using StatsBase
using DataFrames

### LINEAR MODEL

# importing the data
isfile("australian.dat") ||
 download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",
         "australian.dat" )
rawdata = readdlm("australian.dat");

df = DataFrames.DataFrame(rawdata,:auto)
rename!(df,:x15 => :class)
df[!,:x4] = [x == 1 ? 1.0 : 0.0 for x in df[!,:x4]]
df[!,:x12] = [x == 1 ? 1.0 : 0.0 for x in df[!,:x12]]
df[!,:x14] = log.(df[!,:x14])
first(df,5)

describe(df)

# count values
countmap(df[!, :class])

# train and test StatsBase
train_ratio = 0.7
train_set = df[1:floor(Int,size(df,1)*train_ratio),:];
test_set = df[floor(Int,size(df,1)*train_ratio + 1):end,:];

# the data is transposed - we are using algebra
X_train = Matrix(train_set[:,1:end-1])';
X_test = Matrix(test_set[:,1:end-1])';
y_train = train_set[!, :class];
y_test = test_set[!, :class];

# data normalization - you must normalise the data if you want to train the nn
function scale(X)
    μ = mean(X, dims=2) #mu
    σ = std(X, dims=2)

    X_norm = (X .- μ) ./ σ
    return (X_norm, μ, σ);
end

# my data is already split, so I do it this way:
function scale(X, μ, σ)
    X_norm = (X .- μ) ./ σ
    return X_norm;
end
# Julia allows us to create functions with the same names with methods for different types of data

# starting with the train set bc it is bigger
X_train, μ, σ = scale(X_train);
X_test = scale(X_test, μ, σ);


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

# approximating the derivative numeratively is risky bc of the underflow problem
# (f(x + Δx) - f(x - Δx)/2Δx
# (0.000000001 - 0.0000000000000211) / 0.00000000000003233233323


# gradient symbol - nabla

# To optimize the weights β i use the simple gradient descent method:

# machine epsilon - smallest possible number on the computer
eps()
# i use epsilon to represent the Δx
√eps()
(f(x + x√eps()) - f(x - x√eps()))/2x√eps()

# step by step:
β
LinearIndices(β)
(LinearIndices(β) .== 2)
β .+ (LinearIndices(β) .== 2) * β[2] * √eps()


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
plt.savefig("mygraph.png")

accuracy(β, X, y, T = 0.5) = sum((Predict(β, X)' .≥ T ).== y)/length(y)
accuracy(β, X_test, y_test)