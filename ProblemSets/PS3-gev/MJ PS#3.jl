
#ECON 6343: Problem Set 3
# Name: Mahendra Jamankar

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, HTTP, Optim, GLM, LinearAlgebra, Random, Statistics, FreqTables
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function allwrap()

# Question:1

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Now perform MNL
function mnl_loglikelihood(β, X, Z, y)
    n, p = size(X)
    J = size(Z, 2)
    γ = β[end]
    β = reshape(β[1:end-1], p, J-1)
    
    ll = 0.0
    for i in 1:n
        denom = 1 + sum(exp.(X[i,:]' * β[:, j] + γ * (Z[i, j] - Z[i, J])) for j in 1:J-1)
        if y[i] < J
            ll += X[i,:]' * β[:, y[i]] + γ * (Z[i, y[i]] - Z[i, J]) - log(denom)
        else
            ll += -log(denom)
        end
    end
    return -ll  # We minimize the negative log-likelihood
end

# Initial parameter guess
β_init = vcat(vec(zeros(size(X, 2), size(Z, 2) - 1)), 0)

# Optimize
result = optimize(β -> mnl_loglikelihood(β, X, Z, y), β_init, BFGS())

# Extract results
β_mnl = Optim.minimizer(result)
println("Multinomial Logit Estimates:")
println(β_mnl)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question:2
#= for a 1% increase in the wage ratio between two occupations, the log-odds of 
choosing the higher-paying occupation change by almost 10%.    =#


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question:3

function nested_logit_loglikelihood(θ, X, Z, y, nest_structure)
    n, p = size(X)
    J = size(Z, 2)
    
    β_WC = θ[1:p]
    β_BC = θ[p+1:2p]
    λ_WC = θ[2p+1]
    λ_BC = θ[2p+2]
    γ = θ[end]
    
    ll = 0.0
    for i in 1:n
        V_WC = sum(exp((X[i,:]' * β_WC + γ * (Z[i, j] - Z[i, J])) / λ_WC) for j in nest_structure["WC"])
        V_BC = sum(exp((X[i,:]' * β_BC + γ * (Z[i, j] - Z[i, J])) / λ_BC) for j in nest_structure["BC"])
        
        denom = 1 + V_WC^λ_WC + V_BC^λ_BC
        
        if y[i] in nest_structure["WC"]
            ll += (X[i,:]' * β_WC + γ * (Z[i, y[i]] - Z[i, J])) / λ_WC + (λ_WC - 1) * log(V_WC) - log(denom)
        elseif y[i] in nest_structure["BC"]
            ll += (X[i,:]' * β_BC + γ * (Z[i, y[i]] - Z[i, J])) / λ_BC + (λ_BC - 1) * log(V_BC) - log(denom)
        else
            ll += -log(denom)
        end
    end
    return -ll
end

# Define nest structure
nest_structure = Dict(
    "WC" => [1, 2, 3],
    "BC" => [4, 5, 6, 7],
    "Other" => [8]
)

# Initial parameter guess
θ_init = vcat(zeros(2*size(X, 2)), [1.0, 1.0], 0)

# Optimize
result_nested = optimize(θ -> nested_logit_loglikelihood(θ, X, Z, y, nest_structure), θ_init, BFGS())

# Extract results
β_nl = Optim.minimizer(result_nested)
println("\nNested Logit Estimates:")
println(β_nl)

end

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question:4

# allwrap()

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question:5
# AI write unit tests for the functions in the code.

using Test

@testset "Multinomial Logit Tests" begin
    # Test with dummy data
    X_test = [1.0 0.0; 0.0 1.0]
    Z_test = [1.0 2.0; 3.0 4.0]
    y_test = [1, 2]
    β_test = [0.5, -0.5, 0.1]

    ll = mnl_loglikelihood(β_test, X_test, Z_test, y_test)
    @test typeof(ll) == Float64
    @test !isnan(ll)
    @test !isinf(ll)
end

@testset "Nested Logit Tests" begin
    # Test with dummy data
    X_test = [1.0 0.0; 0.0 1.0]
    Z_test = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y_test = [1, 3]
    θ_test = [0.5, -0.5, 0.3, -0.3, 1.1, 1.2, 0.1]
    nest_structure_test = Dict("WC" => [1, 2], "BC" => [3])

    ll = nested_logit_loglikelihood(θ_test, X_test, Z_test, y_test, nest_structure_test)
    @test typeof(ll) == Float64
    @test !isnan(ll)
    @test !isinf(ll)
end

