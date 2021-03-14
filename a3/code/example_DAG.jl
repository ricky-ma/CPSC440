using JLD
include("bayesian.jl")

data = load("../data/onTime.jld")
X = round.(Int64,data["X"])

thetas = MLE(X)
println(thetas)

samples = generateSamples(thetas,10000,4)
sampledThetas = MLE(samples)
error = round.(abs.(thetas .- sampledThetas), digits=3)
println(sampledThetas)
println("Error between MLEs from actual data vs. sampled data:",error)

# using true model, p(0,1,0,1) should equal approx. 0.0004
instances = (samples[:,1].==0) .& (samples[:,2].==1) .& (samples[:,3].==0) .& (samples[:,4].==1)
p0101 = count(instances) / length(samples)
println("p(0,1,0,1) from samples:",p0101)
