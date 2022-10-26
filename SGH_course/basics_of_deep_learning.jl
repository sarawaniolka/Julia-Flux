# Classification task on the Australian credit scoring data. 
# Goal: Distinguisz between the good and bad debtors

using DelimitedFiles
using PyPlot
using Statistics
using StatsBase
using DataFrames

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

  