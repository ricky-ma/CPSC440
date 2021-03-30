using Printf, LinearAlgebra

function blogreg(X,y,lambda,nSamples)

	(n,d) = size(X);

	samples = zeros(nSamples,d)

	# Initialize and compute negative log-posterior (up to constant)
	w = zeros(d,1);
	log_p = logisticObj(w,X,y,lambda)

	nAccept = 0
	for s in 1:nSamples

		# Propose candidate
		wHat = w + randn(d,1)
		log_phat = logisticObj(wHat,X,y,lambda)

		# Metropolis-Hastings accept/reject step (in log-domain)
		logR = log_phat - log_p
		if log(rand()) < logR
			w = wHat
			log_p = log_phat
			nAccept += 1
			@printf("Accepted sample %d, acceptance rate = %f\n",s,nAccept/s)
		end
		samples[s,:] = w
	end

	return samples
end


function logisticObj(w,X,y,lambda)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	return -f -(lambda/2)*dot(w,w)
end
