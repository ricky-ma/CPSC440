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

function leastSquaresEmpiricalBasis(x,y)
	λs = [0.01, 0.02, 0.03, 0.04, 0.05]
	σs = [0.5, 0.75, 1, 1.25, 1.5]
	ps = [1, 2, 3, 4, 5]

	maxlikelihood = -Inf
	best = [0, 0, 0]
	for p in ps
		Z = polyBasis(x,p)
		n,k = size(Z)
		for λ in λs
			for σ in σs
				Θ = (1/σ^2) * (Z'*Z) + λ*I  # posterior precision
				w = (1/σ^2) * inv(Θ) * (Z'*y)  # posterior mean
				# marginal likelihood
				c = (λ^(.5k)) / (((σ*sqrt(2*pi))^n)*det(Θ)^(.5))
				l = log(c) - (1/(2σ^2))*norm(Z*w-y) - .5λ*norm(w)
				# update hyperparameters
				if l > maxlikelihood
					best = [p,σ,λ]
					maxlikelihood = l
				end
			end
		end
	end

	println(best)
	p,σ,λ = best
	p = Int(p)
	Z = polyBasis(x,p)
	Θ = (1/σ^2) * (Z'*Z) + λ*I  # posterior precision
	w = (1/σ^2) * inv(Θ) * (Z'*y)  # posterior mean
	predict(xhat) = polyBasis(xhat,p)*w
	return GenericModel(predict)
end
