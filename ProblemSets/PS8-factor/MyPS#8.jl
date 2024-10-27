#ECON 6343: PS#8
#Name: Mahendra Jamankar

#**************************************************************************
# Install MultivariateStats if not already installed
using Pkg
Pkg.add("MultivariateStats")

# Load all required packages
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using MultivariateStats
#*****************************************************************************


#*************************************
# Question1: linear regression model
#*************************************

# Download and load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Fit the linear regression model
model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)

# Display the regression results
println("Regression Results:")
println("==================")
println(model)

# Get detailed statistics
coef_table = coeftable(model)
println("\nDetailed Statistics:")
println("==================")
println(coef_table)

# Calculate R-squared
r2 = r²(model)
println("\nR-squared: ", round(r2, digits=4))


#*************************************
# Question2: corr. among asvab vars. 
#*************************************

# Extract ASVAB variables
asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
asvab_data = Matrix(df[:, asvab_vars])

# Calculate correlation matrix
cor_matrix = cor(asvab_data)

# Print formatted correlation matrix
println("\nCorrelation Matrix of ASVAB Variables:")
println("----------------------------------------")

# Print header row
print("         ")
for var in asvab_vars
    print(rpad(String(var)[7:end], 8))  # Remove 'asvab' prefix for cleaner output
end
println()

# Print correlation matrix with row labels
for (i, row_var) in enumerate(asvab_vars)
    print(rpad(String(row_var)[7:end], 8))  # Remove 'asvab' prefix
    for j in 1:length(asvab_vars)
        print(rpad(round(cor_matrix[i,j], digits=3), 8))
    end
    println()
end


#*************************************
# Question3: factor analysis
#*************************************

# Fit the regression model with ASVAB variables
model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
           asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)

# Display regression results
println("Regression Results:")
println("==================")
println(model)

# Get detailed statistics
coef_table = coeftable(model)
println("\nDetailed Statistics:")
println("==================")
println(coef_table)

# Calculate R-squared
r2 = r²(model)
println("\nR-squared: ", round(r2, digits=4))

# Calculate VIF (Variance Inflation Factors) to check for multicollinearity
function calculate_vif(X)
    n, p = size(X)
    vif = zeros(p)
    for i in 1:p
        other_cols = setdiff(1:p, i)
        Xi = X[:, i]
        Xothers = X[:, other_cols]
        model_i = lm(Xothers, Xi)
        r2_i = r²(model_i)
        vif[i] = 1 / (1 - r2_i)
    end
    return vif
end

# Prepare matrix for VIF calculation
X = Matrix(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr, 
           :asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])

# Calculate VIF
vif_values = calculate_vif(X)
println("\nVariance Inflation Factors:")
println("==========================")
vars = ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", 
        "asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
for (var, vif) in zip(vars, vif_values)
    println(rpad(var, 10), " : ", round(vif, digits=2))
end

#= Comment on multicollinearity: Yes, including all six ASVAB variables directly in the regression is problematic. 
The high correlations between ASVAB variables we observed earlier suggest strong multicollinearity, which can:inflate standard errors,
make coefficient estimates unstable, make individual effects harder to isolate. 
The VIF values for the ASVAB variables confirm the presence of multicollinearity (VIF > 5 or 10 is often considered problematic)=#


#*************************************
# Question4: first principle component
#*************************************

# Extract ASVAB variables and create matrix
asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
asvabMat = Matrix(df[:, asvab_vars])'  # Transpose to get J×N matrix

# Fit PCA and get first principal component
M = fit(PCA, asvabMat; maxoutdim=1)
asvabPCA = MultivariateStats.transform(M, asvabMat)

# Reshape PCA scores and add to dataframe
df.asvab_pc1 = reshape(asvabPCA, :)

# Fit regression model with first principal component
model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_pc1), df)

# Display regression results
println("Regression Results:")
println("==================")
println(model)

# Get detailed statistics
coef_table = coeftable(model)
println("\nDetailed Statistics:")
println("==================")
println(coef_table)

# Calculate R-squared
r2 = r²(model)
println("\nR-squared: ", round(r2, digits=4))

# Calculate proportion of variance explained by first PC
println("\nVariance Explained by First Principal Component:")
println("=============================================")
println(round(principalratio(M), digits=4))


#*************************************
# Question5: Factor Analysis
#*************************************

# Fit Factor Analysis and get first factor
M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabFA = MultivariateStats.transform(M, asvabMat)

# Reshape factor scores and add to dataframe
df.asvab_f1 = reshape(asvabFA, :)

# Fit regression model with first factor
model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_f1), df)

# Display regression results
println("Regression Results:")
println("==================")
println(model)

# Get detailed statistics
coef_table = coeftable(model)
println("\nDetailed Statistics:")
println("==================")
println(coef_table)

# Calculate R-squared
r2 = r²(model)
println("\nR-squared: ", round(r2, digits=4))

# Display factor loadings
println("\nFactor Loadings:")
println("===============")
loadings = M.W  # Extract loadings
for (var, loading) in zip(asvab_vars, loadings)
    println(rpad(String(var)[7:end], 8), " : ", round(loading, digits=4))
end


#*************************************
# Question6: 
#*************************************


# Define measurement system functions
function create_design_matrices()
    # X_m matrix (measurement equation covariates)
    X_m = hcat(ones(nrow(df)), df.black, df.hispanic, df.female)
    
    # X matrix (wage equation covariates)
    X = hcat(ones(nrow(df)), df.black, df.hispanic, df.female, 
             df.schoolt, df.gradHS, df.grad4yr)
    
    # ASVAB scores
    M = Matrix(df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])
    
    return X_m, X, M, df.logwage
end

function get_gaussian_quadrature(n_points=15)
    # Get Gauss-Legendre quadrature points and weights
    nodes, weights = gausslegendre(n_points)
    # Transform from [-1,1] to [-6,6] for better coverage of normal distribution
    nodes = nodes .* 6
    weights = weights .* 6
    return nodes, weights
end

function normal_pdf(x)
    return exp(-0.5 * x^2) / sqrt(2π)
end

function individual_likelihood(θ, X_m_i, X_i, M_i, y_i, nodes, weights)
    # Unpack parameters
    n_asvab = 6
    α = reshape(θ[1:(4*n_asvab)], 4, n_asvab)  # Measurement eq parameters
    β = θ[(4*n_asvab+1):(4*n_asvab+7)]        # Wage eq parameters
    γ = θ[(4*n_asvab+8):(4*n_asvab+13)]       # Factor loadings
    δ = θ[4*n_asvab+14]                        # Wage eq factor loading
    σ_j = θ[(4*n_asvab+15):(4*n_asvab+20)]    # Measurement error SDs
    σ_w = θ[end]                               # Wage equation error SD
    
    # Initialize likelihood
    likelihood = 0.0
    
    # Gauss-Legendre quadrature
    for (node, weight) in zip(nodes, weights)
        ξ = node
        
        # Measurement equations likelihood
        meas_ll = 0.0
        for j in 1:n_asvab
            μ_j = X_m_i' * α[:, j] + γ[j] * ξ
            meas_ll += log(normal_pdf((M_i[j] - μ_j) / σ_j[j]) / σ_j[j])
        end
        
        # Wage equation likelihood
        μ_w = X_i' * β + δ * ξ
        wage_ll = log(normal_pdf((y_i - μ_w) / σ_w) / σ_w)
        
        # Combine and weight by quadrature weight and standard normal density
        likelihood += exp(meas_ll + wage_ll) * weight * normal_pdf(ξ)
    end
    
    return likelihood
end

function log_likelihood(θ, X_m, X, M, y, nodes, weights)
    ll = 0.0
    for i in 1:size(X_m, 1)
        X_m_i = X_m[i, :]
        X_i = X[i, :]
        M_i = M[i, :]
        y_i = y[i]
        
        l_i = individual_likelihood(θ, X_m_i, X_i, M_i, y_i, nodes, weights)
        ll += log(max(l_i, 1e-10))  # Prevent log(0)
    end
    return -ll  # Return negative for minimization
end


#*************************************
# Question7: Unit test
#*************************************

using Test
Pkg.add("FastGaussQuadrature")
using FastGaussQuadrature

# Helper function to create mock data
function create_mock_data()
    Random.seed!(42)
    n = 100  # number of observations
    
    # Create mock DataFrame
    df = DataFrame(
        logwage = randn(n),
        black = rand([0, 1], n),
        hispanic = rand([0, 1], n),
        female = rand([0, 1], n),
        schoolt = rand(8:16, n),
        gradHS = rand([0, 1], n),
        grad4yr = rand([0, 1], n),
        asvabAR = randn(n),
        asvabCS = randn(n),
        asvabMK = randn(n),
        asvabNO = randn(n),
        asvabPC = randn(n),
        asvabWK = randn(n)
    )
    return df
end

@testset "Statistical Analysis Tests" begin
    df = create_mock_data()
    
    @testset "VIF Calculation" begin
        # Test VIF calculation function
        X = Matrix(df[:, [:black, :hispanic, :female]])
        vif_values = calculate_vif(X)
        
        @test length(vif_values) == 3  # Should return VIF for each variable
        @test all(vif_values .>= 1.0)  # VIF values should be >= 1
    end
    
    @testset "Gaussian Quadrature" begin
        nodes, weights = get_gaussian_quadrature(5)
        
        @test length(nodes) == length(weights)
        @test length(nodes) == 5
        @test sum(weights) ≈ 12.0  # Should sum to approximately the interval width ([-6,6] = 12)
    end
    
    @testset "Normal PDF" begin
        @test normal_pdf(0.0) ≈ 1/sqrt(2π)
        @test normal_pdf(1.0) ≈ exp(-0.5)/sqrt(2π)
        @test normal_pdf(-1.0) ≈ exp(-0.5)/sqrt(2π)
    end
    
    @testset "Design Matrices Creation" begin
        X_m, X, M, y = create_design_matrices()
        
        @test size(X_m, 2) == 4  # Should have intercept + 3 demographic variables
        @test size(X, 2) == 7    # Should have intercept + 6 covariates
        @test size(M, 2) == 6    # Should have 6 ASVAB variables
        @test length(y) == size(X, 1)  # Should have same number of observations
    end
    
    @testset "Individual Likelihood" begin
        # Create small test case
        X_m_i = ones(4)
        X_i = ones(7)
        M_i = ones(6)
        y_i = 1.0
        nodes, weights = get_gaussian_quadrature(3)
        
        # Mock parameters
        θ = vcat(
            ones(24),  # α (4×6)
            ones(7),   # β
            ones(6),   # γ
            1.0,       # δ
            ones(6),   # σ_j
            1.0        # σ_w
        )
        
        ll = individual_likelihood(θ, X_m_i, X_i, M_i, y_i, nodes, weights)
        @test ll > 0  # Likelihood should be positive
    end
end