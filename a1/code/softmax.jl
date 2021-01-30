include("findMin.jl")

function softmaxObj(w,X,y,k)
	# Calculate the function value
	(n,d) = size(X)
	W = reshape(w,k,d)
	# Calculate the function value
	f = 0
	for i in 1:n
		f += (-X*W')[i,y[i]] + log(sum(exp.(X*W'), dims=2)[i])
	end
	# Calculate the gradient value
	g = zeros(k,d)
	for i in 1:n
		p = exp.((X*W')[i,:]) ./ sum(exp.(X*W'),dims=2)[i]
		for c in 1:k
			g[c,:] += X[i,:]*(p[c] - (y[i]==c))
		end
	end
	g = reshape(g,k*d,1) # flatten
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)
	# Each column of 'w' will be a logistic regression classifier
	W = zeros(k,d)
	funObj(w) = softmaxObj(w,X,y,k)
	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=100)
	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W',dims=2)
	return LinearModel(predict,W)
end
