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


function sampleRejection(p1,pt,numSamples,queryTime,condTime,endState)
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
            if t == condTime && state == endState
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


function viterbiDecode(p1,pt,d)
    k = length(p1)
    indices = Array{Int32}(undef,d,k)
    optimalDecoding = Array{Int32}(undef,d-1)

    # forward pass: store probabilities for each timestep
    bestCase = p1
    for t in 2:d
        # enumerate all possible transitions
        possibleTransitions = bestCase' .* pt
        # choose max for each state and store in bestCase
        bestCase,inds = findmax(possibleTransitions; dims=2)

        # TODO: fix indexing so that backtracking works correctly
        # save argmax indices to indices
        indices[t-1,:] = [i[2] for i in inds]
    end

    # backtracking: find most likely sequence from the end
    _, highestP = findmax(bestCase)
    highestP = highestP[1]
    for t=reverse(1:d-1)
        optimalDecoding[t] = highestP
        highestP = indices[t,highestP]
    end
    println(optimalDecoding)
    return optimalDecoding
end
