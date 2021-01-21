include("misc.jl")
using LinearAlgebra

function leastSquares(X,y)
	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function rbfBasis(Xtilde,X,sigma)
	# Add bias column
	ntilde = size(Xtilde,1)
	Ztilde = [ones(ntilde,1) Xtilde]
	n = size(X,1)
	Z = [ones(n,1) X]
	# Calculate kernal using Gaussian radial basis function
	K = exp.(-distancesSquared(Ztilde, Z)./(2*sigma*sigma))
	return K
end

function leastSquaresRBFL2(X,y,lambda,sigma)
	# Find regression weights w/ L2 regularization
	K = rbfBasis(X,sigma)
	w = (K'*K + lambda*I)\(K'*y)
	# Make linear prediction function
	predict(Xtilde) = rbfBasis(Xtilde,X,sigma)*w
	# Return model
	return LinearModel(predict,w)
end
