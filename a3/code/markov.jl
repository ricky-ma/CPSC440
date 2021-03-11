using StatsBase
include("misc.jl")

function sampleAncestral(p1,pt,d,numSamples)
    k = length(p1)
    samples = Array{Float64}(undef,k,numSamples)
    # generate samples for Monte Carlo estimation
    for s in 1:numSamples
        # sample ancestrally for d timesteps
        stateDist = p1
        for t in 1:d
            stateDist = pt[sampleDiscrete(stateDist),:]
        end
        samples[:,s] = stateDist
    end
    # Monte Carlo estimation
    estimate = Array{Float64}(undef,k)
    for j in 1:k
        # compute marginal by summing probabilities of each column
        estimate[j] = sum(samples[j,:])
    end
    # normalize and return estimate
    return estimate ./ numSamples
end


function sampleRejection(p1,pt,numSamples,queryTime,condTime,condState)
    # Monte Carlo estimation via rejection sampling:
    # estimate frequency of first event in samples consistent with second event
    frequencies = [0,0,0,0,0,0,0]
    acceptedSamples = 0
    # generate samples for Monte Carlo estimation
    for s in 1:numSamples
        # flags for rejection sampling
        accept = false
        acceptState = 0
        # sample ancestrally until endTime
        stateDist = p1
        for t in 1:max(queryTime,condTime)
            state = sampleDiscrete(stateDist)
            stateDist = pt[state,:]
            # if t=queryTime, keep track of state
            if t == queryTime
                acceptState = state
            end
            # if t=endTime and we are at desired endState, accept sample
            if t == condTime && state == condState
                accept = true
                acceptedSamples += 1
            end
        end
        if accept
            frequencies[acceptState] += 1
        end
    end
    conditionalProbabilities = frequencies ./ acceptedSamples
    return conditionalProbabilities, acceptedSamples
end


function marginalCK(p1,pt,d)
    mostLikelyStates = Array{Int32}(undef,d)
    pd = p1
    for t in 1:d-1
        val,ind = findmax(pd)
        mostLikelyStates[t] = ind
        pd = pt' * pd
    end
    val,ind = findmax(pd)
    mostLikelyStates[d] = ind
    return pd, mostLikelyStates
end


function conditionalCK(p1,pt,queryTime,condTime,condState)
    # get marginal probabilities up to query time and conditional time
    pQuery, _ = marginalCK(p1,pt,queryTime)
    pCond, _ = marginalCK(p1,pt,condTime)
    k = length(p1)
    # DP: iteratively find conditional probabilities given marginals
    condProbs = zeros(k)
    for j in 1:k
        currState = zeros(k)
        currState[j] = 1
        pd, _ = marginalCK(currState,pt,condTime)
        condProbs[j] = (pd[condState] * pQuery[j]) / pCond[condState]
    end
    return condProbs
end


function viterbiDecode(p1,pt,d)
    k = length(p1)
    M = zeros(2,d,k)
    optimalDecoding = Array{Int32}(undef,d)
    # forward pass: store probabilities for each timestep
    # initilize first best-case as p1
    M[1,1,:] = p1
    for t in 2:d
        # for each state, enumerate all possible transitions
        for j in 1:k
            possibleTransitions = M[1,t-1,:] .* pt[:,j]
            # find max for each transition and store in M
            M[:,t,j] .= findmax(possibleTransitions)
        end
    end
    # backtracking: find most likely sequence from the end
    optimalDecoding[d] = findmax(M[1,d,:])[2]
    for t=reverse(2:d)
        optimalDecoding[t-1] = M[2,t,optimalDecoding[t]]
    end
    return optimalDecoding
end


function homogeneousMarkovMLEs(Xtrain)
    p = proportions(Xtrain[:,1])
    numStates = length(unique(Xtrain[:,1]))
    theta = Array{Float64}(undef,numStates,numStates)

    # shift data by 1 for 1-day time-series analysis
    Xcurr = Xtrain[:,1:d-1]
    Xnext = Xtrain[:,2:d]
    # total number of sunny and rainy days
    numSunny = count(Xcurr .== 0)
    numRainy = count(Xcurr .== 1)

    # p(xt+1=sunny|xt=sunny)
    theta[1,1] = count(Xnext[Xcurr .== 0] .== 0) / numSunny
    # p(xt+1=rainy|xt=sunny)
    theta[1,2] = count(Xnext[Xcurr .== 0] .== 1) / numSunny
    # p(xt+1=sunny|xt=rainy)
    theta[2,1] = count(Xnext[Xcurr .== 1] .== 0) / numRainy
    # p(xt+1=rainy|xt=rainy)
    theta[2,2] = count(Xnext[Xcurr .== 1] .== 1) / numRainy
    return p, theta
end

function homogeneousMarkovNLL(Xvalid,p,theta)
    n,d = size(Xvalid)
    NLL = 0
    # get log-likelihood (l) at each timestep for each sample
    # l = sum_j^d log(p(Xi|Xi-1)) = log(p(Xi=xi)) + sum_ij(theta_ij)
    # NLL = -log(p(Xi=xi)) - sum_ij(theta_ij)
    for i in 1:n
        # +1 to adjust for 1-based indexing
        NLL -= log(p[Xvalid[i,1] + 1])
        for j in 2:d
            currState = Xvalid[i,j] + 1
            prevState = Xvalid[i,j-1] + 1
            NLL -= log(theta[prevState,currState])
        end
    end
    return NLL
end
