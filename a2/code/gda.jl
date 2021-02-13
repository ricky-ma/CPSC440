using Statistics, LinearAlgebra, StatsBase, ProgressBars, Plots

function logPDF(xi,pic,mu,Sigma)
    deter = -0.5 * logdet(Sigma)
    expon = -0.5 * ((xi - mu)'*inv(Sigma)*(xi - mu))
    p = log(pic) .+ expon .+ deter
    return p[1][1]
end

function PDF(xi,pic,mu,Sigma)
    (d,) = size(mu)
    temp = 1/((2*pi)^(d/2)*det(Sigma)^(1/2))
    p = temp * exp(-((Sigma \ xi)' * xi) ./2)
    return p[1] * pic
end

function gdaPredict(Xhat,k,mus,Sigmas,pis)
    n,d = size(Xhat)
    xhatMu = mean(Xhat, dims=1)
    yhat = Array{Float64}(undef,n)
    for (i,xi) in enumerate(eachrow(Xhat))
        p = Array{Float64}(undef,k)
        for c in 1:k
            p[c] = logPDF(xi - xhatMu', pis[c], mus[c,:], Sigmas[c,:,:])
        end
        yhat[i] = findmax(p)[2]
    end
    return yhat
end

function gda(X,y)
    # Filter and density estimation to get mu and Sigma for each class
    k = length(unique(y))
    n,d = size(X)
    mus = Array{Float64}(undef,k,d)
    Sigmas = Array{Float64}(undef,k,d,d)
    pis = Array{Float64}(undef,k)
    # For each class c, center data and calculate parameters:
    # prior pi, mean mu, covariance sigma
    for c in 1:k
        Xc = X[y.==c,:] .- mean(X, dims=1)
        pis[c] = size(Xc)[1]/n
        mus[c,:] = mean(Xc, dims=1)
        Sigmas[c,:,:] = cov(Xc)
    end
    # Prediction on Xhat
    predict(Xhat) = gdaPredict(Xhat,k,mus,Sigmas,pis)
    return GenericModel(predict)
end

function responsibilities(X,pis,mus,Sigmas,t,k)
    r = Array{Float64}(undef,t,k)
    for i in 1:t
        pxy = Array{Float64}(undef,k)
        for c in 1:k
            pxy[c] = PDF(X[i,:], pis[c], mus[c,:], Sigmas[c,:,:])
        end
        r[i,:] = pxy ./ sum(pxy)
    end
    return r
end

# E-step: expectation of complete log-likelihood given last parameters Theta^t
function Qfunction(X,y,Xtest,pis,mus,Sigmas,r,n,t,k)
    q = 0
    for i in 1:n
        q += logPDF(X[i,:],pis[y[i]],mus[y[i],:],Sigmas[y[i],:,:])
    end
    for i in 1:t
        for c in 1:k
            q += r[i,c] * logPDF(Xtest[i,:],pis[c],mus[c,:],Sigmas[c,:,:])
        end
    end
    return q
end

# M-step: update parameters Theta^t+1 given last parameters Theta^t
function emUpdate(X,y,Xtest,pis,mus,Sigmas,r,n,t,k,ncs)
    for c in 1:k
        Xc = X[y.==c,:]
        pis[c] = (ncs[c] + sum(r[:,c])) / n+t
        muSums = sum(Xc, dims=1) + sum(r[:,c].*Xtest, dims=1)
        mus[c,:] = muSums ./ (ncs[c] + sum(r[:,c]))
        varSums = (Xc' .- mus[c,:]) * (Xc' .- mus[c,:])'
        varSums += r[:,c]' .* (Xtest' .- mus[c,:]) * (Xtest' .- mus[c,:])'
        Sigmas[c,:,:] = varSums ./ (ncs[c] + sum(r[:,c]))
    end
    return pis,mus,Sigmas
end

function gdaSSL(X,y,Xtest;maxIters=50,epsilon=1e-5)
    k = length(unique(y))
    n,d = size(X)
    t,_ = size(Xtest)
    mus = Array{Float64}(undef,k,d)
    Sigmas = Array{Float64}(undef,k,d,d)
    pis = ones(k)./k
    ncs = counts(y,k)
    # Center data and initialize mus and Sigmas
    X = X .- mean(X,dims=1)
    Xtest = Xtest .- mean(Xtest,dims=1)
    for c in 1:k
        mus[c,:] = mean(X, dims=1)
        Sigmas[c,:,:] = cov(X)
    end
    # Expectation maximization
    Q = -Inf
    Qs = Array{Float64}(undef,maxIters)
    for iter in ProgressBar(1:maxIters)
        r = responsibilities(Xtest,pis,mus,Sigmas,t,k)
        Qt = Qfunction(X,y,Xtest,pis,mus,Sigmas,r,n,t,k) # E-step
        pis,mus,Sigmas = emUpdate(X,y,Xtest,pis,mus,Sigmas,r,n,t,k,ncs) # M-step
        Qs[iter] = Qt
        if Qt - Q <= epsilon
            println("Iterations: $iter\nQt-Q=$(Qt-Q)")
            break
        end
        Q = Qt
    end
    display(plot((1:maxIters),Qs))
    predict(Xhat) = gdaPredict(Xhat,k,mus,Sigmas,pis)
    return GenericModel(predict)
end
