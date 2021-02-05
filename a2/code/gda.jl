using Statistics

function gda(X, y)
    # Filter and density estimation to get mu and Sigma for each class
    labels = 1:length(unique(y))
    X_with_labels = [y X]
    mus = Array{Float64}(undef,0,12)
    Sigmas = []
    pis = []
    for label in 1:length(labels)
        # filter data by class label and remove label column
        Xc = X_with_labels[X_with_labels[:,1] .== label, :][:, 1:end .!= 1]
        pic = length(Xc)/length(X)
        pis = push!(pis, pic)
        # mean vector containing mean of each feature, by label
        mu = mean(Xc, dims=1)
        mus = [mus; mu]
        # calculate Sigma for this class
        # Sigma = sum((Xc .- mu)*(Xc .- mu)') ./ length(Xc)
        Sigma = cov(Xc)
        Sigmas = push!(Sigmas, Sigma)
    end

    function calculate_pdf(mu, Sigma, pic, xi)
        log_variance = log(Sigma)
        deter = -0.5 * logdet(inv(Sigma))
        expon = -0.5 * ((xi' .- mu)*inv(Sigma)*(xi' .- mu)')
        p = log(pic) .+ expon .+ deter
        return p[1][1]
    end

    function predict(Xhat)
        yhat = Vector{Int32}()
        for xi in eachrow(Xhat)
            probabilities = Vector{Float64}()
            for c in 1:length(labels)
                p = calculate_pdf(mus[c], Sigmas[c], pis[c], xi)
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
