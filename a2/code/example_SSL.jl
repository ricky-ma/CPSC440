
using JLD, Statistics, Printf

# Load X and y variable
data = load("SSL.jld")
(X,y,Xtest,ytest,Xbar) = (data["X"],data["y"],data["Xtest"],data["ytest"],data["Xbar"])

# Fit a KNN classifier
k = 1
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

include("gda.jl")
model = gda(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with GDA: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with GDA: %.3f\n",testError)

model = gdaSSL(X,y,Xtest)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with semi-supervised GDA: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with semi-supervised GDA: %.3f\n",testError)
