# Load X and y variable
using JLD

# Load initial probabilities and transition probabilities of Markov chain
data = load("../data/gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

# Confirm that initial probabilities sum up to 1
@show sum(p1)

# Confirm that transition probabilities sum up to 1, starting from each state
@show sum(pt,dims=2)

include("markov.jl")

estimate = sampleAncestral(p1,pt,50,10000)
println()
println("Univariate marginals at time 50 using a Monte Carlo estimate based on 10000 samples:")
println(round.(estimate, digits=3))

exact = marginalCK(p1,pt,50)
println("Exact univariate marginals at time 50 using CK equations:")
println(round.(exact, digits=3))

println("Error between approximation and exact:")
println(round.(abs.(estimate - exact), digits=3))
