include("misc.jl")

function sampleAncestral(p1,pt,steps,numSamples)
    k = length(p1)
    samples = Array{Float64}(undef,k,numSamples)
    # generate samples for Monte Carlo estimation
    for s in 1:numSamples
        # sample ancestrally
        state = p1
        for t in 1:steps
            state = pt[sampleDiscrete(state),:]
        end
        samples[:,s] = state
    end
    # Monte Carlo estimation
    estimate = Array{Float64}(undef,k)
    for j in 1:k
        # compute marginal by summing probabilities of each column
        estimate[j] = sum(samples[j,:])
    end
    # normalize and return estimate
    estimate = estimate ./ numSamples
    return estimate
end


function marginalCK(p1,pt,steps)
    pd = p1
    for t in 1:steps-1
        pd = pt' * pd
    end
    return pd
end


function viterbiDecode(p1,pt,steps)

end
