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

function Qfunction(X,y,Xtest,pis,mus,Sigmas,r,n,t,k)
    q = 0
    for i in 1:n
        c = y[i]
        q += logPDF(X[i,:],pis[c],mus[c,:],Sigmas[c,:,:])
    end
    for i in 1:t
        for c in 1:k
            q += r[i,c] + logPDF(Xtest[i,:],pis[c],mus[c,:],Sigmas[c,:,:])
        end
    end
    return q
end

function gdaSSL(X,y,Xtest;maxIters=100,epsilon=1e-4)
    k = length(unique(y))
    n,d = size(X)
    t,_ = size(Xtest)
    mus = Array{Float64}(undef,k,d)
    Sigmas = Array{Float64}(undef,k,d,d)
    pis = Array{Float64}(undef,k)
    ncs = counts(y,k)
    # Center data
    X = X .- mean(X,dims=1)
    Xtest = Xtest .- mean(Xtest,dims=1)

    # for each class c:
    # calculate parameters: prior theta, mean mu, covariance sigma
    for c in 1:k
        Xc = X[y.==c,:]
        pis[c] = size(Xc)[1]/n
        mus[c,:] = mean(Xc, dims=1)
        Sigmas[c,:,:] = cov(Xc)
    end

    Q = 0
    Qs = Array{Float64}(undef,maxIters)
    for iter in ProgressBar(1:maxIters)
        r = responsibilities(Xtest,pis,mus,Sigmas,t,k)
        for c in 1:k
            Xc = X[y.==c,:]
            # parameter updates
            pis[c] = (ncs[c] + sum(r[:,c])) / n+t
            muSums = sum(X[y.==c,:], dims=1) + sum(r[:,c].*Xtest, dims=1)
            mus[c,:] = muSums ./ (ncs[c] + sum(r[:,c]))
            varSums = (Xc' .- mus[c,:]) * (Xc' .- mus[c,:])'
            varSums += r[:,c]' .* (Xtest' .- mus[c,:]) * (Xtest' .- mus[c,:])'
            Sigmas[c,:,:] = varSums ./ (ncs[c] + sum(r[:,c]))
        end

        Qt = Qfunction(X,y,Xtest,pis,mus,Sigmas,r,n,t,k)
        Qs[iter] = Qt
        # if Qt - Q <= epsilon
        #     println(Qt-Q)
        #     break
        # end
        Q = Qt
    end
    display(plot((1:maxIters),Qs))
    # Prediction on Xhat
    predict(Xhat) = gdaPredict(Xhat,k,mus,Sigmas,pis)
    return GenericModel(predict)
end
