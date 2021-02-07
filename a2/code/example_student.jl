using LinearAlgebra, PyPlot

# Generate data from a Gaussian with outlier
n = 250
d = 2
nOutliers = 25
mu = randn(d)
Sigma = randn(d,d)
Sigma = (1/2)*(Sigma+Sigma') # Make symmetric
sd = eigen(Sigma)
Sigma += (1-minimum(sd.values))*I # Make positive-definite
R = cholesky(Sigma) # Get a matrix acting like the "standard deviation": Sigma = A*A'
A = R.L
X = zeros(n,d)
for i in 1:n
    xi = randn(d) # Sample from multivariate standard normal
    X[i,:] = A*xi + mu # Sample from multivariate Gaussian (by affine property)
end
X[rand(1:n,nOutliers),:] = abs.(10*rand(nOutliers,d)) # Add some crazy points

include("studentT.jl")
model = studentT(X)

# Plot data and densities (you can ignore the code below)
plot(X[:,1],X[:,2],".")

increment = 100
(xmin,xmax) = xlim()
xDomain = range(xmin,stop=xmax,length=increment)
(ymin,ymax) = ylim()
yDomain = range(ymin,stop=ymax,length=increment)

xValues = repeat(xDomain,1,length(xDomain))
yValues = repeat(yDomain',length(yDomain),1)
z = model.pdf([xValues[:] yValues[:]])
@assert(length(z) == length(xValues),"Size of model function's output is wrong");
zValues = reshape(z,size(xValues))
contour(xValues,yValues,zValues)

# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a TDA classifier
model = tda(Xtest,ytest,10)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with TDA classifier: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with TDA classifier: %.3f\n",testError)
