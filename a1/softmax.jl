
function softmaxObj(w,X,y,k)
	# Calculate the function value
	(n,d) = size(X)
    w = zeros(k*d)
	w = reshape(w, (k,d))
	f = 0
	for i in 1:n
		f += dot(-w[y[i]],X[i]) + log(sum(exp.(w*X[i]')))
	end
	# Calculate the gradient value
	g = zeros((k,d))
	for c in 1:k
		for j in 1:d
			for i in 1:n
				p = exp.(dot(w[c],X[i])) / sum(exp.(w*X[i]'))
				I = y[i] == c
				g[c][j] += X[i][j] * (p-I)
			end
		end
	end
	g = Iterators.flatten(g)
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = softmaxObj(w,X,y,k)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end
