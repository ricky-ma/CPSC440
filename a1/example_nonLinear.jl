# Load X and y variable
using JLD
using Random
include("leastSquares.jl")
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Split X and y into training and validation sets
partition = 0.5
idx = shuffle(1:n)
train_ind = view(idx, 1:floor(Int, partition*n))
val_ind = view(idx, (floor(Int, partition*n)+1):n)
Xtrain = X[train_ind,:]
Xval = X[val_ind,:]
ytrain = y[train_ind,:]
yval = y[val_ind,:]

best = [Inf, 0, 0]
# Perform simple grid-search to determine hyperparameters
lambdas = [0:0.005:0.1;]
sigmas = [0.1:0.1:1;]
for lambda in lambdas
    for sigma in sigmas
        # Train model
        model = leastSquaresRBFL2(Xtrain,ytrain,lambda,sigma)
        # Report the error on the test set
        using Printf
        t = size(Xtest,1)
        yhat = model.predict(Xval)
        valError = sum((yhat - yval).^2)/t
        @printf("Validation Error = %.2f\n",valError)
        # Update best error
        if valError < best[1]
            best = [valError, lambda, sigma]
        end
    end
end
@printf("Lowest validation error = %.2f\n",best[1])
@printf("lambda = %.2f\n",best[2])
@printf("sigma = %.2f\n",best[3])

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
model = leastSquaresRBFL2(Xtrain,ytrain,2,0.3)
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
gcf()
savefig("q2_2_plot.png")
