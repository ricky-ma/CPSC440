using JLD
include("bayesian.jl")

data = load("../data/onTime.jld")
X = round.(Int64,data["X"])

thetas = MLE(X)
println(thetas)

# TODO: fix sample generation to account for dependencies
samples = generateSamples(thetas,100000)
thetas = MLE(samples)
println(thetas)

# using true model, p(0,1,0,1) should equal approx. 0.0004
p0101 = count(samples[:,2] + samples[:,4] .== 2) / length(samples)
println(p0101)
