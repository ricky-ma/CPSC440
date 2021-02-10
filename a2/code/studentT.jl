include("misc.jl")
include("findMin.jl")
using SpecialFunctions

function studentT(X)
    (n,d) = size(X)
    # Initialize parameters
    mu = zeros(d)
    Sigma = Matrix(Diagonal(ones(d)))
    dof = 3*ones(1,1)

    # Optimize by block coordinate optimization
    mu_old = ones(d)
    funObj_mu(mu) = NLL(X,mu,Sigma,dof,1)
    funObj_Sigma(Sigma) = NLL(X,mu,Sigma,dof,2)
    funObj_dof(dof) = NLL(X,mu,Sigma,dof,3)
    while norm(mu-mu_old,Inf) > .1
    #for i in 1:5
        mu_old = mu
        # Update mean
        mu = findMin(funObj_mu,mu,verbose=false)
        # Update covariance
        Sigma[:] = findMin(funObj_Sigma,Sigma[:],verbose=false)
        # Update degrees of freedom
        dof = findMin(funObj_dof,dof,verbose=false)
    end

    # Compute square root of log-determinant
    SigmaInv = Sigma^-1
    R = cholesky(Sigma)
    A = R.L
    logStd = sum(log.(diag(A)))
    dof = dof[1] # Take scalar out of containiner
    logZ = logabsgamma((dof+d)/2)[1] - (d/2)*log(pi) - logStd - logabsgamma(dof/2)[1] - (d/2)*log(dof)

    function pdf(Xhat)
        (t,d) = size(Xhat)
        PDFs = zeros(t)
        for i in 1:t
            xCentered = Xhat[i,:] - mu
            tmp = 1 + (1/dof)*dot(xCentered,SigmaInv*xCentered)
            nll = ((d+dof)/2)*log(tmp) - logZ
            PDFs[i] = exp(-nll)
        end
        return PDFs
    end
    return DensityModel(pdf)
end


function NLL(X,mu,Sigma,dof,deriv)
    (n,d) = size(X)
    # Initialize nll and gradient
    nll = 0
    if deriv == 1
        g = zeros(d,1)
    elseif deriv == 2
        g = zeros(d,d)
    else
        g = zeros(1,1)
    end
    # Possibly re-shape and symmeetrize sigma
    if size(Sigma,1) != d
        Sigma = reshape(Sigma,d,d)
    end
    Sigma = (Sigma+Sigma')/2
    dof = dof[1]
    if dof < 0
        return (Inf,g[:])
    end

    # Compute a square root of Sigma, and check that it's positive-definite
    A = zeros(d,d)
    try
        R = cholesky(Sigma)
        A = R.L
    catch
        # Case where Sigma is not positive-definite
        return (Inf,g[:])
    end

    # Take into account log factor
    SigmaInv = Sigma^-1
    for i in 1:n
        xCentered = X[i,:] - mu
        tmp = 1 + (1/dof)*dot(xCentered,SigmaInv*xCentered)
        nll += ((d+dof)/2)*log(tmp)
        if deriv == 1
            g -= ((d+dof)/(dof*tmp))*SigmaInv*xCentered
        elseif deriv == 2
            g -= ((d+dof)/(2*dof*tmp))*SigmaInv*xCentered*xCentered'*SigmaInv
        else
            g[1] -= (d/(2*tmp*dof^2))*dot(xCentered,SigmaInv*xCentered)
            g[1] -= (dof/(2*tmp*dof^2))*dot(xCentered,SigmaInv*xCentered)
            g[1] += (1/2)*log(tmp)
        end
    end

    # Now take into account normalizing constant
    logStd = sum(log.(diag(A)))
    logZ = logabsgamma((dof+d)/2)[1] - (d/2)*log(pi) - logStd - logabsgamma(dof/2)[1] - (d/2)*log(dof)
    nll -= n*logZ
    if deriv == 2
        g += (n/2)*SigmaInv
        g = (g+g')/2
        g = g[:]
    elseif deriv == 3
        g[1] += - (n/2)*polygamma(0,(dof+d)/2) + (n/2)*polygamma(0,dof/2) + n*(d/(2*dof))
    end
    return (nll,g)
end

function tda(X, y, k)
    X_with_labels = [y X]
    subModels = Array{DensityModel}(undef,k)
    priors = Array{Float64}(undef,k)
    for c in 1:k
        # filter data by class label, remove label column, and center
        Xc = X_with_labels[X_with_labels[:,1] .== c, :][:, 1:end .!= 1]
        # prior probability pi_c
        priors[c] = length(Xc)/length(X)
        # create and store density model for this specific class
        subModels[c] = studentT(Xc)
    end

    function predict(Xhat)
        n,d = size(Xhat)
        # calculate pdfs for each class
        probabilties = Array{Float64}(undef,n,k)
        for c in 1:k
            pdfs = subModels[c].pdf(Xhat)
            probabilties[:,c] = pdfs .* priors[c]
        end
        # for each xi, yi = index of largest pdf
        yhat = Array{Int32}(undef,n)
        for (i,row) in enumerate(eachrow(probabilties))
            val, yi = findmax(row)
            yhat[i] = yi
        end
        return yhat
    end
    return GenericModel(predict)
end
