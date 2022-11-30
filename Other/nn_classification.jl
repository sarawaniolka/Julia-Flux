# based on https://medium.com/coffee-in-a-klein-bottle/deep-learning-with-julia-e7f15ad5080b

# Simple classification Neural Network using Flux

# imports
using Plots
using Statistics
using Flux

# data
function real_data(n)
    x1 = rand(1, n) .- 0.5
    x2 = (x1 .* x1)*3 .+ randn(1, n) * 0.1
    return vcat(x1, x2)
end

function fake_data(n)
    θ  = 2*π*rand(1,n)
    r  = rand(1,n)/3
    x1 = @. r*cos(θ)
    x2 = @. r*sin(θ)+0.5
    return vcat(x1,x2)
end

# dataset
train_size = 5000;
real = real_data(train_size)
fake = fake_data(train_size)

# visualization
scatter(real[1, 1:500], real[2, 1:500])
scatter!(fake[1, 1:500], fake[2, 1:500])
# blue - real, orange - fake

# Neural Network
function nn()
    return Chain(
        Dense(2, 25, relu), 
        Dense(25, 1, x->σ.(x)))
end

# organizing the data in batches
X = hcat(real, fake)
Y = vcat(ones(train_size), zeros(train_size))
data = Flux.Data.DataLoader((X, Y'), batchsize = 100, shuffle=true)

# The model
m = nn()

# optimization function
opt = Descent(0.05) # gradient descent

# loss function
loss(x, y) = sum(Flux.logitbinarycrossentropy(m(x), y)) # cross - entropy

# Training the model
params = Flux.params(m)
epochs = 500;
for i in 1:epochs
    Flux.train!(loss, params, data, opt)
end

# print the results
println(mean(m(real))," ", mean(m(fake)))
scatter(real[1,1:100],real[2,1:100],zcolor=m(real)')
scatter!(fake[1,1:100],fake[2,1:100],zcolor=m(fake)',legend=false)


# for epochs = 50 the results are not too good, the look like horizontal layers
# for epochs = 500 they look quite good

# Training method 2
m = nn()
function trainModel!(m,data;epochs=200)
    for epoch = 1:epochs
        for d in data
            gs = gradient(Flux.params(m)) do
                l = loss(d...)
            end
            Flux.update!(opt, Flux.params(m), gs)
        end
    end
    @show mean(m(real)),mean(m(fake))
end
trainModel!(m,data;epochs=200)

scatter(real[1,1:100],real[2,1:100],zcolor=m(real)')
scatter!(fake[1,1:100],fake[2,1:100],zcolor=m(fake)',legend=false)