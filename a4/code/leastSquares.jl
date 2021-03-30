using LinearAlgebra
include("misc.jl")


function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	w = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*w

	return GenericModel(predict)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

