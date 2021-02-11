using Statistics, LinearAlgebra

function gda(X,y)
    # Filter and density estimation to get mu and Sigma for each class
    k = length(unique(y))
    d = size(X)[2]
    mus = Array{Float64}(undef,k,d)
    Sigmas = Array{Float64}(undef,k,d,d)
    thetas = Array{Float64}(undef,k)
    globalMu = mean(X, dims=1)
    for c in 1:k
        # filter data by class label and center
        Xc = X[y.==c,:] .- globalMu
        # calculate parameters: prior theta, mean mu, covariance sigma
        thetas[c] = length(Xc)/length(X)
        mus[c,:] = mean(Xc, dims=1)
        Sigmas[c,:,:] = cov(Xc)
    end

    function PDF(mu, Sigma, theta, xi)
        deter = -0.5 * logdet(Sigma)
        expon = -0.5 * ((xi - mu)'*inv(Sigma)*(xi - mu))
        p = log(theta) .+ expon .+ deter
        return p[1][1]
    end

    function predict(Xhat)
        xhatMu = mean(Xhat, dims=1)
        yhat = Vector{Int32}()
        for xi in eachrow(Xhat)
            probabilities = Vector{Float64}()
            for c in 1:k
                p = PDF(mus[c,:], Sigmas[c,:,:], thetas[c], xi - xhatMu')
                append!(probabilities, p)
            end
            # Use PDF and responsibility to get categorical distribution
            yi = findmax(probabilities)[2]
            append!(yhat, yi)
        end
        return yhat
    end
    return GenericModel(predict)
end

function gdaSSL(X,y)
    gdaModel = gda(X,y)
    mus = gdaModel.mus
    Sigmas = gdaModel.Sigmas
    thetas = gdaModel.thetas
    println("hello")
end
