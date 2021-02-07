using Statistics

function gda(X, y)
    # Filter and density estimation to get mu and Sigma for each class
    k = length(unique(y))
    d = size(X)[2]
    X_with_labels = [y X]
    mus = Array{Float64}(undef,k,d)
    Sigmas = Array{Float64}(undef,k,d,d)
    pis = Array{Float64}(undef,k)
    globalMu = mean(X, dims=1)
    for c in 1:k
        # filter data by class label, remove label column, and center
        Xc = X_with_labels[X_with_labels[:,1] .== c, :]
        Xc = Xc[:, 1:end .!= 1] .- globalMu
        pis[c] = length(Xc)/length(X)
        # mean vector containing mean of each feature, by label
        mu = mean(Xc, dims=1)
        mus[c,:] = mu
        # calculate Sigma for this class
        # println(dot((Xc[c,:]' - mu),(Xc[c,:]' - mu)'))
        # Sigma = (Xc[:,c] - mu)*(Xc[:,c] - mu)' ./ length(Xc)
        # println(size(Sigma))
        Sigmas[c,:,:] = cov(Xc)
    end

    function PDF(mu, Sigma, pic, xi)
        deter = -0.5 * logdet(Sigma)
        expon = -0.5 * ((xi - mu)'*inv(Sigma)*(xi - mu))
        p = log(pic) .+ expon .+ deter
        return p[1][1]
    end

    function predict(Xhat)
        xhatMu = mean(Xhat, dims=1)
        yhat = Vector{Int32}()
        for xi in eachrow(Xhat)
            probabilities = Vector{Float64}()
            for c in 1:k
                p = PDF(mus[c,:], Sigmas[c,:,:], pis[c], xi - xhatMu')
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
