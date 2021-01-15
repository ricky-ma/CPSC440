using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

include("PCA.jl")
# model = PCA(X,2)
# Z = model.compress(X)
V = cov(X)
values, vectors = eigen(V)
explained_variance = values ./ sum(values)
show(findmax(explained_variance))

# Show scatterplot of 2 features
figure(2)
clf()
plot(Z,".")
for i in rand(1:n,20)
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end
gcf()
savefig("scatterplot.png")
