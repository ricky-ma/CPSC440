using LinearAlgebra
include("misc.jl")

function MLE(X)
    n,d = size(X)
    thetas = Array{Float64}(undef,d)
    # i.e. x1=alarm, x2=latebus, x3=overslept, x4=ontime
    #           latebus -> ontime
    # alarm -> overslept -> ontime

    # alarm and latebus don't have parents, so just divide by n
    thetas[1] = count(X[:,1] .== 1) / n
    thetas[2] = count(X[:,2] .== 1) / n
    # overslept depends on alarm, so divide count(overslept,alarm) by count(alarm)
    thetas[3] = count(X[:,1] + X[:,2] .== 2) / count(X[:,1] .== 1)
    # ontime depends on latebus and overslept, so divide by count(overslept,latebus)
    thetas[4] = count(X[:,1] + X[:,2] + X[:,3] .== 3) / count(X[:,2] + X[:,3] .== 2)
    return thetas
end


function generateSamples(thetas,numSamples)
    d = length(thetas)
    samples = Array{Float64}(undef,numSamples,d)
    for i in 1:numSamples
        sample = Array{Int32}(undef,d)
        # for j in 1:d
        #     p = [1-thetas[j], thetas[j]]
        #     sample[j] = sampleDiscrete(p) - 1
        # end

        p1 = [1-thetas[1], thetas[1]]
        sample[1] = sampleDiscrete(p1) - 1

        p2 = [1-thetas[2], thetas[2]]
        sample[2] = sampleDiscrete(p2) - 1

        p3 = [1-thetas[3], thetas[3]]
        sample[3] = sampleDiscrete(p3) - 1

        p4 = [1-thetas[4], thetas[4]]
        sample[4] = sampleDiscrete(p4) - 1
        samples[i,:] = sample
    end
    return samples
end
