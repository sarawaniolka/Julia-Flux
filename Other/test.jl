begin
    import Pkg;
    packages = ["CSV","DataFrames","PlutoUI","Plots","Combinatorics"]   
    Pkg.add(packages)

    using CSV, DataFrames, PlutoUI, Plots, Combinatorics

    plotly()
    theme(:solarized_light)
end

begin
    df = CSV.read("iris.csv", DataFrame; normalizenames=true);
    dropmissing!(df)

end

begin
    df_species = groupby(df, :variety)
end

begin
    Pkg.add("Flux")
    Pkg.add("CUDA")
    Pkg.add("IterTools")

    using Flux
    using Flux: Data.DataLoader
    using Flux: @epochs
    using CUDA
    using Random
    using IterTools: ncycle

    Random.seed!(123);
end

begin   
    # Convert df to array
    data = convert(Array, df)

    # Shuffle
    data = data[shuffle(1:end), :]

    # train/test split
    train_test_ratio = .7
    idx = Int(floor(size(df, 1) * train_test_ratio))
    data_train = data[1:idx,:]
    data_test = data[idx+1:end, :]

    # Get feature vectors
    get_feat(d) = transpose(convert(Array{Float32},d[:, 1:end-1]))
    x_train = get_feat(data_train)
    x_test = get_feat(data_test)

    # One hot labels
    #   onehot(d) = [Flux.onehot(v, unique(df.class)) for v in d[:,end]]
    onehot(d) = Flux.onehotbatch(d[:,end], unique(df.variety))
    y_train = onehot(data_train)
    y_test = onehot(data_test)
end

begin
    batch_size= 1
    train_dl = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
    test_dl = DataLoader((x_test, y_test), batchsize=batch_size)
end

begin
    ### Model ------------------------------
    function get_model()
        c = Chain(
            Dense(4,8,relu),
            Dense(8,3),
            softmax
        )
    end

    model = get_model()

    ### Loss ------------------------------
    loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y)

    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []

    ### Optimiser ------------------------------
    lr = 0.001
    opt = ADAM(lr, (0.9, 0.999))

    ### Callbacks ------------------------------
    function loss_all(data_loader)
        sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader) 
    end

    function acc(data_loader)
        f(x) = Flux.onecold(cpu(x))
        acces = [sum(f(model(x)) .== f(y)) / size(x,2)  for (x,y) in data_loader]
        sum(acces) / length(data_loader)
    end

    callbacks = [
        () -> push!(train_losses, loss_all(train_dl)),
        () -> push!(test_losses, loss_all(test_dl)),
        () -> push!(train_acces, acc(train_dl)),
        () -> push!(test_acces, acc(test_dl)),
    ]

    # Training ------------------------------
    epochs = 30
    ps = Flux.params(model)

    @epochs epochs Flux.train!(loss, ps, train_dl, opt, cb = callbacks)

    @show train_loss = loss_all(train_dl)
    @show test_loss = loss_all(test_dl)
    @show train_acc = acc(train_dl)
    @show test_acc = acc(test_dl)
end 


y = (y_test[:,1])
pred = (model(x_test[:,1]))
x_axis = 1:epochs * size(y_train,2)
plot(x_axis, train_losses, label = "Training loss", title = "Loss", xaxis = "epochs * data size")