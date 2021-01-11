using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features
j1 = rand(1:d)
j2 = rand(1:d)
figure(2)
clf()
plot(X[:,j1],X[:,j2],".")
for i in rand(1:n,10)
    annotate(dataTable[i+1,1],
	xy=[X[i,j1],X[i,j2]],
	xycoords="data")
end
