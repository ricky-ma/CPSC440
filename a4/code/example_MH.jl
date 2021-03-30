# Load X and y variable
using JLD, PyPlot
data = load("twoThrees.jld")
(X,y) = (data["X"],data["y"])

include("blogreg.jl")
nSamples = 10000
samples = blogreg(X,y,1,nSamples)

figure(1);
j=1;
pos = [119 144 152 162 184 196];
neg = [30 75 106 109 123 124];
neu = [79 167 182 213 222 255];
for i in 1:6
    global j
    subplot(6,3,j);
    j=j+1;
    hist(samples[:,pos[i]]);
    title("Positive Variables");
    subplot(6,3,j);
    j=j+1;
    hist(samples[:,neg[i]]);
    title("Negative Variables");
    subplot(6,3,j);
    j=j+1;
    hist(samples[:,neu[i]]);
    title("Neutral Variables");
end
