using JLD
include("markov.jl")

# Load initial probabilities and transition probabilities of Markov chain
data = load("../data/rain.jld")
X = round.(Int64,data["X"])
(n,d) = size(X)

# Split into a training and validation set
splitNdx = Int(ceil(n/2))
trainNdx = 1:splitNdx
validNdx = splitNdx+1:n
Xtrain = X[trainNdx,:]
Xvalid = X[validNdx,:]
nTrain = length(trainNdx)
nValid = length(validNdx)

# Fit a single Bernoulli to the entire dataset
theta = sum(Xtrain .== 1)/(nTrain*d)

# Measure test set NLL
NLL = 0
for i in 1:nValid
    for j in 1:d
        if Xvalid[i,j] == 1
            global NLL -= log(theta)
        else
            global NLL -= log(1-theta)
        end
    end
end
@show NLL
println()

p,theta = homogeneousMarkovMLEs(Xtrain)
NLL = homogeneousMarkovNLL(Xvalid,p,theta)
println("pi:",round.(p, digits=3))
println("theta:",round.(theta, digits=3))
println("NLL:",NLL)
