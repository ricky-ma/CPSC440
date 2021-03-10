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
exact, mostLikelyStates = marginalCK(p1,pt,50)
println()
println("Univariate marginals at time 50 using a Monte Carlo estimate based on 10000 samples:")
println(round.(estimate, digits=3))
println("Exact univariate marginals at time 50 using CK equations:")
println(round.(exact, digits=3))
println("Error between approximation and exact:")
println(round.(abs.(estimate - exact), digits=3))
println()
println("State with highest marginal probability for each time j:")
println(mostLikelyStates)
println()

# decoding = viterbiDecode(p1,pt,100)
# println()

pgrad = [0,0,1,0,0,0,0]
exact, mostLikelyStates = marginalCK(pgrad,pt,50)
println("p(x50=c|x1=3):")
println(round.(exact, digits=3))
println()

println("p(x5=c|x10=6):")
estimate = sampleRejection(p1,pt,10000,5,10,6)
println(round.(estimate, digits=3))
