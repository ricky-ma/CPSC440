include("misc.jl")

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

function rbfBasis(X,sigma)
	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]
	# Calculate kernal using Gaussian radial basis function
	distances_squared = distancesSquared(Z, Z)
	K = exp.(-distances_squared./(2*sigma*sigma))
	return K
end

function leastSquaresRBFL2(X,y,lambda,sigma)
	# Find regression weights
	K = rbfBasis(X,y,sigma)
	w = (K'*K)\(K'*y)

	# Make linear prediction function and add L2 regularization
	regularization = lambda*sum(w.^2)/2
	# predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w + regularization
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w .+ regularization

	# Return model
	return LinearModel(predict,w)
end
