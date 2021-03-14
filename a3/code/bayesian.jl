include("misc.jl")

function MLE(X)
    n,d = size(X)
    # x1=alarm, x2=latebus, x3=overslept, x4=ontime
    #           latebus -> ontime
    # alarm -> overslept -> ontime

    # alarm and latebus don't have parents, so just divide by n
    pa1 = count(X[:,1] .== 1) / n
    pb1 = count(X[:,2] .== 1) / n
    # overslept depends on alarm, so divide count(overslept,alarm) by count(alarm)
    # do for alarm=True and alarm=False
    ps1a0 = count((X[:,1].==0) .& (X[:,3].==1)) / count(X[:,1] .== 0)
    ps1a1 = count((X[:,1].==1) .& (X[:,3].==1)) / count(X[:,1] .== 1)
    # ontime depends on latebus and overslept, so divide by count(latebus,overslept)
    # do for (latebus=1,overslept=1), (latebus=0,overslept=1), (latebus=1,overslept=0), (latebus=0,overslept=0)
    po1b1s1 = count((X[:,2].==1) .& (X[:,3].==1) .& (X[:,4].==1)) / count((X[:,2].==1) .& (X[:,3].==1))
    po1b0s1 = count((X[:,2].==0) .& (X[:,3].==1) .& (X[:,4].==1)) / count((X[:,2].==0) .& (X[:,3].==1))
    po1b1s0 = count((X[:,2].==1) .& (X[:,3].==0) .& (X[:,4].==1)) / count((X[:,2].==1) .& (X[:,3].==0))
    po1b0s0 = count((X[:,2].==0) .& (X[:,3].==0) .& (X[:,4].==1)) / count((X[:,2].==0) .& (X[:,3].==0))
    return pa1,pb1,ps1a0,ps1a1,po1b1s1,po1b0s1,po1b1s0,po1b0s0
end


function generateSamples(thetas,numSamples,d)
    samples = Array{Float64}(undef,numSamples,d)
    for i in 1:numSamples
        # sample alarm
        pa = [1-thetas[1], thetas[1]]
        samples[i,1] = sampleDiscrete(pa) - 1

        # sample latebus
        pb = [1-thetas[2], thetas[2]]
        samples[i,2] = sampleDiscrete(pb) - 1

        # sample overslept, conditioned on alarm
        poa = Array{Float64}(undef,2)
        if samples[i,1] == 0
            poa = [1-thetas[3], thetas[3]]
        elseif samples[i,1] == 1
            poa = [1-thetas[4], thetas[4]]
        end
        samples[i,3] = sampleDiscrete(poa) - 1

        # sample onTime, conditioned on latebus and overslept
        pobs = Array{Float64}(undef,2)
        if samples[i,2] == 1 && samples[i,3] == 1
            pobs = [1-thetas[5], thetas[5]]
        elseif samples[i,2] == 0 && samples[i,3] == 1
            pobs = [1-thetas[6], thetas[6]]
        elseif samples[i,2] == 1 && samples[i,3] == 0
            pobs = [1-thetas[7], thetas[7]]
        elseif samples[i,2] == 0 && samples[i,3] == 0
            pobs = [1-thetas[8], thetas[8]]
        end
        samples[i,4] = sampleDiscrete(pobs) - 1
    end
    return samples
end
