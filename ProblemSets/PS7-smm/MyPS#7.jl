#:::::::::::::::::::::::::::::::::::::::::::::::::
# ECON 6343: Econometrics III
# Problem Set 7
# Name: Mahendra Jamankar
#:::::::::::::::::::::::::::::::::::::::::::::::::

using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using Distributions

#::::::::::::::::::::::::
# Question 1
#::::::::::::::::::::::::

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Clean and prepare the data
df = dropmissing(df, :occupation)
df[df.occupation .∈ Ref([8,9,10,11,12,13]), :occupation] .= 7

# Create feature matrix and response vector
X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
y = df.married .== 1

# For multinomial logit
y_mult = df.occupation

# Define the moment function
function g(β, X, y)
    ε = y - X * β
    return X' * ε
end

# Define the GMM objective function
function gmm_objective(β, X, y, W)
    moment = g(β, X, y)
    return moment' * W * moment
end

# Estimate using GMM
n, k = size(X)
W = I(k)  # Identity matrix as weighting matrix
β_init = rand(k)

result_gmm = optimize(β -> gmm_objective(β, X, y, W), β_init, LBFGS())
β_gmm = result_gmm.minimizer

# Compare with closed-form OLS
β_ols = inv(X'X) * X'y

println("GMM estimates: ", β_gmm)
println("OLS estimates: ", β_ols)

#::::::::::::::::::::::::
# Question 2
#::::::::::::::::::::::::

# (a) Maximum likelihood estimation:
function mlogit_ll(β, X, y)
    n, k = size(X)
    J = maximum(y)
    β_matrix = reshape(β, k, J-1)
    
    ll = 0.0
    for i in 1:n
        denom = 1 + sum(exp.(X[i,:] .* β_matrix))
        if y[i] < J
            ll += X[i,:] ⋅ β_matrix[:,y[i]] - log(denom)
        else
            ll += -log(denom)
        end
    end
    return -ll
end

β_init = rand(size(X,2) * (maximum(y_mult)-1))
result_mle = optimize(β -> mlogit_ll(β, X, y_mult), β_init, LBFGS())
β_mle = result_mle.minimizer

println("MLE estimates: ", β_mle)


# Define the multinomial logit probability function
function mlogit_probs(β, X)
    n, k = size(X)
    J = Int(length(β) / k) + 1
    β_matrix = reshape(β, k, J-1)
    
    probs = zeros(n, J)
    for i in 1:n
        denom = 1 + sum(exp.(X[i,:] .* β_matrix))
        for j in 1:(J-1)
            probs[i,j] = exp(X[i,:] ⋅ β_matrix[:,j]) / denom
        end
        probs[i,J] = 1 / denom
    end
    return probs
end

# Define the moment function for multinomial logit
function g_mlogit(β, X, y)
    n, k = size(X)
    J = Int(length(β) / k) + 1
    probs = mlogit_probs(β, X)
    
    g = zeros(n * J)
    for i in 1:n
        for j in 1:J
            idx = (i-1)*J + j
            g[idx] = (y[i] == j ? 1 : 0) - probs[i,j]
        end
    end
    return X' * reshape(g, n, J)
end

# Define the GMM objective function
function gmm_objective_mlogit(β, X, y, W)
    moment = vec(g_mlogit(β, X, y))
    return moment' * W * moment
end

# Set up the weighting matrix
W = I(size(X,2) * maximum(y_mult))  # Identity matrix as weighting matrix

# (b) GMM estimation with MLE starting values
result_gmm_mle = optimize(β -> gmm_objective_mlogit(β, X, y_mult, W), β_mle, LBFGS())
β_gmm_mle = result_gmm_mle.minimizer

# (c) GMM estimation with random starting values
β_init_random = rand(length(β_mle))
result_gmm_random = optimize(β -> gmm_objective_mlogit(β, X, y_mult, W), β_init_random, LBFGS())
β_gmm_random = result_gmm_random.minimizer

# Compare results
println("GMM estimates (MLE start): ", β_gmm_mle)
println("GMM estimates (random start): ", β_gmm_random)

# Compare objective function values
obj_mle_start = gmm_objective_mlogit(β_gmm_mle, X, y_mult, W)
obj_random_start = gmm_objective_mlogit(β_gmm_random, X, y_mult, W)

println("Objective function value (MLE start): ", obj_mle_start)
println("Objective function value (random start): ", obj_random_start)

# Check if the estimates are close
is_close = isapprox(β_gmm_mle, β_gmm_random, rtol=1e-5)
println("Are the estimates close? ", is_close)

# Comment on global concavity
if is_close && isapprox(obj_mle_start, obj_random_start, rtol=1e-5)
    println("The objective function appears to be globally concave, as both starting values led to similar estimates and objective function values.")
else
    println("The objective function may not be globally concave, as different starting values led to different estimates or objective function values.")
end


#::::::::::::::::::::::::
# Question 3
#::::::::::::::::::::::::

# Function to simulate multinomial logit data
function simulate_multinomial_logit(N, J, K, β_true)
    X = hcat(ones(N), randn(N, K-1))  # Design matrix with constant term
    U = X * reshape(β_true, K, J-1)   # Utility matrix
    U = hcat(U, zeros(N))             # Add zero utility for base choice
    P = exp.(U) ./ sum(exp.(U), dims=2)  # Choice probabilities
    Y = [rand(Categorical(P[i,:])) for i in 1:N]  # Simulate choices
    return X, Y
end

# Multinomial logit log-likelihood function
function mlogit_ll(β, X, Y)
    N, K = size(X)
    J = maximum(Y)
    U = X * reshape(β, K, J-1)
    U = hcat(U, zeros(N))
    P = exp.(U) ./ sum(exp.(U), dims=2)
    ll = sum(log(P[i, Y[i]]) for i in 1:N)
    return -ll
end

# Set parameters
N = 10000  # Number of observations
J = 4      # Number of choices (> 2)
K = 3      # Number of covariates (> 1, including constant)

# True parameters (K * (J-1) vector)
β_true = [1.0, -0.5, 0.5, -0.5, 1.0, -1.0, 0.5, -1.0, 1.0]

# Simulate data
X, Y = simulate_multinomial_logit(N, J, K, β_true)

# Estimate parameters
β_init = rand(K * (J-1))
result = optimize(β -> mlogit_ll(β, X, Y), β_init, LBFGS())
β_est = result.minimizer

# Debug prints to check dimensions
println("Size of β_est: ", size(β_est))
println("K: ", K, ", J-1: ", J-1)
println("Product K * (J-1): ", K * (J-1))

# Compare true and estimated parameters
println("True parameters:")
display(reshape(β_true, K, J-1))
println("\nEstimated parameters:")

# Ensure the dimensions are consistent before reshaping
if K * (J-1) == length(β_est)
    reshaped_β_est = reshape(β_est, K, J-1)
    display(reshaped_β_est)
else
    println("Error: Inconsistent dimensions for reshaping β_est")
end

# Check if estimates are close
is_close = isapprox(β_true, β_est, rtol=1e-2)
println("\nAre the estimates close to true values? ", is_close)

# Calculate mean absolute error
mae = mean(abs.(β_true - β_est))
println("Mean Absolute Error: ", mae)

# Additional diagnostics
println("\nOptimization result:")
println("Converged: ", Optim.converged(result))
println("Iterations: ", Optim.iterations(result))
println("Minimum value: ", Optim.minimum(result))


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5: Re-estimate multinomial logit from Question 2 using SMM
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Function to simulate multinomial logit data (from Question 3)
function simulate_multinomial_logit(N, J, K, β)
    X = hcat(ones(N), randn(N, K-1))  # Design matrix with constant term
    U = X * reshape(β, K, J-1)   # Utility matrix
    U = hcat(U, zeros(N))             # Add zero utility for base choice
    
    # Compute probabilities with safeguards
    P = exp.(U .- maximum(U, dims=2))  # Subtract maximum for numerical stability
    P = P ./ sum(P, dims=2)  # Normalize
    
    # Ensure no zero probabilities
    P = P .+ 1e-10
    P = P ./ sum(P, dims=2)
    
    Y = [rand(Categorical(P[i,:])) for i in 1:N]  # Simulate choices
    return X, Y
end

# Function to compute moments for SMM
function compute_moments(Y, J)
    N = length(Y)
    moments = zeros(J)
    for j in 1:J
        moments[j] = sum(Y .== j) / N
    end
    return moments
end

# SMM objective function
function smm_objective(θ, X, Y, J, D)
    N, K = size(X)
    β = reshape(θ, K, J-1)
    
    # Compute data moments
    data_moments = compute_moments(Y, J)
    
    # Simulate model and compute model moments
    model_moments = zeros(J, D)
    for d in 1:D
        _, Y_sim = simulate_multinomial_logit(N, J, K, vec(β))
        model_moments[:, d] = compute_moments(Y_sim, J)
    end
    
    # Average model moments across simulations
    avg_model_moments = mean(model_moments, dims=2)
    
    # Compute difference between data and model moments
    moment_diff = data_moments - vec(avg_model_moments)
    
    # SMM objective (identity weighting matrix)
    return sum(moment_diff.^2)  # Return a scalar
end

# Set parameters
N = 10000  # Number of observations
J = 4      # Number of choices (> 2)
K = 3      # Number of covariates (> 1, including constant)
D = 5000     # Number of simulations for SMM

# True parameters (K * (J-1) vector)
β_true = [1.0, -0.5, 0.5, -0.5, 1.0, -1.0, 0.5, -1.0, 1.0]

# Simulate data
X, Y = simulate_multinomial_logit(N, J, K, β_true)

# SMM estimation
β_init = rand(K * (J-1))
result_smm = optimize(θ -> smm_objective(θ, X, Y, J, D), β_init, LBFGS())
β_est_smm = result_smm.minimizer

# Print results
println("True parameters:")
display(reshape(β_true, K, J-1))
println("\nSMM Estimated parameters:")
display(reshape(β_est_smm, K, J-1))

# Check if estimates are close
is_close = isapprox(β_true, β_est_smm, rtol=1e-2)
println("\nAre the SMM estimates close to true values? ", is_close)

# Calculate mean absolute error
mae = mean(abs.(β_true - β_est_smm))
println("Mean Absolute Error: ", mae)

# Additional diagnostics
println("\nOptimization result:")
println("Converged: ", Optim.converged(result_smm))
println("Iterations: ", Optim.iterations(result_smm))
println("Minimum value: ", Optim.minimum(result_smm))


#::::::::::::::::::::::::::
# Question 6: Wrap up code 
#::::::::::::::::::::::::::

# Define the econometrics analysis function
function run_econometrics_analysis()
    # Load and prepare the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref([8,9,10,11,12,13]), :occupation] .= 7
    X = [ones(size(df,1)) df.age df.race.==1 df.collgrad.==1]
    y = df.married .== 1
    y_mult = df.occupation

    # Define GMM and MLE functions
    function g(β, X, y)
        ε = y - X * β
        return X' * ε
    end

    function gmm_objective(β, X, y, W)
        moment = g(β, X, y)
        return moment' * W * moment
    end

    function mlogit_ll(β, X, y)
        n, k = size(X)
        J = maximum(y)
        β_matrix = reshape(β, k, J-1)
        ll = 0.0
        for i in 1:n
            denom = 1 + sum(exp.(X[i,:] .* β_matrix))
            if y[i] < J
                ll += X[i,:] ⋅ β_matrix[:,y[i]] - log(denom)
            else
                ll += -log(denom)
            end
        end
        return -ll
    end

    function mlogit_probs(β, X)
        n, k = size(X)
        J = Int(length(β) / k) + 1
        β_matrix = reshape(β, k, J-1)
        probs = zeros(n, J)
        for i in 1:n
            denom = 1 + sum(exp.(X[i,:] .* β_matrix))
            for j in 1:(J-1)
                probs[i,j] = exp(X[i,:] ⋅ β_matrix[:,j]) / denom
            end
            probs[i,J] = 1 / denom
        end
        return probs
    end

    function g_mlogit(β, X, y)
        n, k = size(X)
        J = Int(length(β) / k) + 1
        probs = mlogit_probs(β, X)
        g = zeros(n * J)
        for i in 1:n
            for j in 1:J
                idx = (i-1)*J + j
                g[idx] = (y[i] == j ? 1 : 0) - probs[i,j]
            end
        end
        return X' * reshape(g, n, J)
    end

    function gmm_objective_mlogit(β, X, y, W)
        moment = vec(g_mlogit(β, X, y))
        return moment' * W * moment
    end

    # GMM estimation
    n, k = size(X)
    W = I(k)
    β_init = rand(k)
    result_gmm = optimize(β -> gmm_objective(β, X, y, W), β_init, LBFGS())
    β_gmm = result_gmm.minimizer

    # OLS estimation
    β_ols = inv(X'X) * X'y

    println("GMM estimates: ", β_gmm)
    println("OLS estimates: ", β_ols)

    # MLE estimation
    β_init = rand(size(X,2) * (maximum(y_mult)-1))
    result_mle = optimize(β -> mlogit_ll(β, X, y_mult), β_init, LBFGS())
    β_mle = result_mle.minimizer

    println("MLE estimates: ", β_mle)

    # GMM estimation with MLE starting values
    W = I(size(X,2) * maximum(y_mult))
    result_gmm_mle = optimize(β -> gmm_objective_mlogit(β, X, y_mult, W), β_mle, LBFGS())
    β_gmm_mle = result_gmm_mle.minimizer

    # GMM estimation with random starting values
    β_init_random = rand(length(β_mle))
    result_gmm_random = optimize(β -> gmm_objective_mlogit(β, X, y_mult, W), β_init_random, LBFGS())
    β_gmm_random = result_gmm_random.minimizer

    println("GMM estimates (MLE start): ", β_gmm_mle)
    println("GMM estimates (random start): ", β_gmm_random)

    obj_mle_start = gmm_objective_mlogit(β_gmm_mle, X, y_mult, W)
    obj_random_start = gmm_objective_mlogit(β_gmm_random, X, y_mult, W)

    println("Objective function value (MLE start): ", obj_mle_start)
    println("Objective function value (random start): ", obj_random_start)

    is_close = isapprox(β_gmm_mle, β_gmm_random, rtol=1e-5)
    println("Are the estimates close? ", is_close)

    if is_close && isapprox(obj_mle_start, obj_random_start, rtol=1e-5)
        println("The objective function appears to be globally concave.")
    else
        println("The objective function may not be globally concave.")
    end
end

# Call the function to run the analysis
run_econometrics_analysis()

#:::::::::::::::::::::::::::::::::::::
# Question 7: AI Unit test
#:::::::::::::::::::::::::::::::::::::

# Unit tests
using Test
@testset "Econometrics Analysis Tests" begin
    # Test g function
    @testset "g function" begin
        X = [1.0 2.0; 3.0 4.0]
        y = [1.0, 2.0]
        β = [0.5, 0.5]
        result = g(β, X, y)
        @test size(result) == (2,)
        @test isapprox(result, [-0.5, -1.5], atol=1e-6)
    end

    # Test gmm_objective function
    @testset "gmm_objective function" begin
        X = [1.0 2.0; 3.0 4.0]
        y = [1.0, 2.0]
        β = [0.5, 0.5]
        W = I(2)
        result = gmm_objective(β, X, y, W)
        @test isa(result, Float64)
        @test result ≥ 0
    end

    # Test mlogit_ll function
    @testset "mlogit_ll function" begin
        X = [1.0 2.0; 3.0 4.0]
        y = [1, 2]
        β = [0.5, -0.5]
        result = mlogit_ll(β, X, y)
        @test isa(result, Float64)
    end

    # Test mlogit_probs function
    @testset "mlogit_probs function" begin
        X = [1.0 2.0; 3.0 4.0]
        β = [0.5, -0.5, 0.5, -0.5]
        result = mlogit_probs(β, X)
        @test size(result) == (2, 3)
        @test all(0 .≤ result .≤ 1)
        @test all(isapprox.(sum(result, dims=2), 1, atol=1e-6))
    end

    # Test simulate_multinomial_logit function
    @testset "simulate_multinomial_logit function" begin
        N, J, K = 1000, 3, 2
        β = [0.5, -0.5, 0.5, -0.5]
        X, Y = simulate_multinomial_logit(N, J, K, β)
        @test size(X) == (N, K)
        @test length(Y) == N
        @test all(1 .≤ Y .≤ J)
    end

    # Test compute_moments function
    @testset "compute_moments function" begin
        Y = [1, 2, 1, 3, 2, 1]
        J = 3
        result = compute_moments(Y, J)
        @test size(result) == (J,)
        @test isapprox(result, [0.5, 0.333333, 0.166667], atol=1e-6)
    end
end

# Wrap-up code to print summary of tests
println("All tests completed.")