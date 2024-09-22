# ECON 6343: Problem Set 4
# Name: Mahendra Jamankar

using DataFrames, CSV, HTTP, Optim, GLM, LinearAlgebra, Random, Statistics, Distributions

#:::::::::::::::::::::::::::::::
# Question 1
#:::::::::::::::::::::::::::::::
# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Function to compute multinomial logit probabilities
function mnl_probs(β, X, Z)
    J = size(Z, 2)
    N = size(X, 1)
    
    # Compute utilities
    V = [X * β[:, j] + Z[:, j] for j in 1:J]
    
    # Compute probabilities
    P = exp.(hcat(V...))
    P = P ./ sum(P, dims=2)
    
    return P
end

# Log-likelihood function
function log_likelihood(θ, X, Z, y)
    J = size(Z, 2)
    β = reshape(θ[1:end-1], :, J-1)
    γ = θ[end]
    
    P = mnl_probs(β, X, Z .- Z[:, J])
    
    # Add column for base alternative
    P = hcat(P, 1 .- sum(P, dims=2))
    
    # Compute log-likelihood
    ll = sum(log.(P[CartesianIndex.(1:size(P,1), y)]))
    
    return -ll  # Negative because we're minimizing
end

# Gradient function using ForwardDiff
function gradient(θ, X, Z, y)
    return ForwardDiff.gradient(θ -> log_likelihood(θ, X, Z, y), θ)
end

# Initial values (you might want to use estimates from PS3 as suggested)
β_init = zeros(size(X, 2), size(Z, 2) - 1)
γ_init = 0.0
θ_init = [vec(β_init); γ_init]

# Optimization
result = optimize(θ -> log_likelihood(θ, X, Z, y), 
                  θ -> gradient(θ, X, Z, y),
                  θ_init, 
                  BFGS(), 
                  Optim.Options(show_trace = true, iterations = 1000))

# Extract results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:end-1], :, size(Z, 2) - 1)
γ_hat = θ_hat[end]

# Compute standard errors
H = ForwardDiff.hessian(θ -> log_likelihood(θ, X, Z, y), θ_hat)
se = sqrt.(diag(inv(H)))

# Print results
println("Estimated coefficients:")
println(β_hat)
println("Estimated γ:")
println(γ_hat)
println("Standard errors:")
println(se)


#:::::::::::::::::::::::::::::::
# Question 2
#:::::::::::::::::::::::::::::::
#The coefficient gamma reflects how utility changes with a 1-unit increase in log wages relative to other occupation
# And coefficient make sense now. 


#::::::::::::::::::::::::::::::::::
# Question 3
#::::::::::::::::::::::::::::::::::

# Function for Gauss-Legendre quadrature
function lgwt(N, a, b)
    N, a, b = promote(N, a, b)
    x = zeros(N)
    w = zeros(N)
    
    m = (N + 1) ÷ 2
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    
    for i = 1:m
        z = cos(π * (i - 0.25) / (N + 0.5))
        
        while true
            p1 = 1.0
            p2 = 0.0
            for j = 1:N
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            end
            pp = N * (z * p1 - p2) / (z * z - 1)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < eps(z)
                break
            end
        end
        
        x[i] = xm - xl * z
        x[N + 1 - i] = xm + xl * z
        w[i] = 2 * xl / ((1 - z * z) * pp * pp)
        w[N + 1 - i] = w[i]
    end
    
    return x, w
end

# Q.1 part(a)
# Practice with quadrature
d = Normal(0, 1)
nodes, weights = lgwt(7, -4, 4)

# Verify integral of density is 1
println("Integral of density:")
println(sum(weights .* pdf.(d, nodes)))

# Verify expectation is 0
println("Expectation:")
println(sum(weights .* nodes .* pdf.(d, nodes)))

# Q.3 part(b)
# Quadrature for N(0, 2) with 7 and 10 points
d2 = Normal(0, 2)
nodes7, weights7 = lgwt(7, -5*2, 5*2)
nodes10, weights10 = lgwt(10, -5*2, 5*2)

println("Variance with 7 points:")
println(sum(weights7 .* nodes7.^2 .* pdf.(d2, nodes7)))

println("Variance with 10 points:")
println(sum(weights10 .* nodes10.^2 .* pdf.(d2, nodes10)))

# Q.3 part (c) 
# Monte Carlo integration
function monte_carlo_integral(f, a, b, D)
    X = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(X))
end

d2 = Normal(0, 2)
a, b = -5*2, 5*2

println("Monte Carlo integration results:")
println("Variance (D=1,000,000):")
println(monte_carlo_integral(x -> x^2 * pdf(d2, x), a, b, 1_000_000))

println("Mean (D=1,000,000):")
println(monte_carlo_integral(x -> x * pdf(d2, x), a, b, 1_000_000))

println("Integral of density (D=1,000,000):")
println(monte_carlo_integral(x -> pdf(d2, x), a, b, 1_000_000))

println("Variance (D=1,000):")
println(monte_carlo_integral(x -> x^2 * pdf(d2, x), a, b, 1_000))


#::::::::::::::::::::::
# Question 4
#::::::::::::::::::::::

# Mixed logit (not run, just structure)
function mixed_logit_ll(θ, X, Z, y, nodes, weights)
    # Extract parameters
    β = reshape(θ[1:end-2], :, size(Z, 2) - 1)
    μ_γ, σ_γ = θ[end-1:end]
    
    # Compute probabilities for each quadrature point
    function compute_probs(γ)
        V = [X * β[:, j] .+ γ * (Z[:, j] .- Z[:, end]) for j in 1:size(Z, 2)-1]
        P = exp.(hcat(V...))
        P = hcat(P, ones(size(P, 1)))
        P = P ./ sum(P, dims=2)
        return P
    end
    
    # Compute likelihood
    ll = 0.0
    for (node, weight) in zip(nodes, weights)
        γ = μ_γ + σ_γ * node
        P = compute_probs(γ)
        ll += weight * sum(log.(P[CartesianIndex.(1:size(P,1), y)]))
    end
    
    return -ll  # Negative because we're minimizing
end

#::::::::::::::::::::::::::::::
# Question 5
#::::::::::::::::::::::::::::::

# Optimization setup for mixed logit (not run)
nodes, weights = lgwt(7, -4, 4)
θ_init_mixed = [vec(β_hat); γ_hat; 1.0]  # Initial σ_γ set to 1.0
result_mixed = optimize(θ -> mixed_logit_ll(θ, X, Z, y, nodes, weights), 
                        θ_init_mixed, 
                        BFGS(), 
                        Optim.Options(show_trace = true, iterations = 1000))

# Extract and print results (not run)
θ_hat_mixed = Optim.minimizer(result_mixed)
β_hat_mixed = reshape(θ_hat_mixed[1:end-2], :, size(Z, 2) - 1)
μ_γ_hat, σ_γ_hat = θ_hat_mixed[end-1:end]

println("Mixed Logit Results:")
println("β_hat:")
println(β_hat_mixed)
println("μ_γ_hat:")
println(μ_γ_hat)
println("σ_γ_hat:")
println(σ_γ_hat)


#:::::::::::::::::::::
# Question 6
#:::::::::::::::::::::

# Wrap all of code into a function

function run_econometrics_analysis()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    # Function to compute multinomial logit probabilities
    function mnl_probs(β, X, Z)
        J = size(Z, 2)
        N = size(X, 1)
        
        # Compute utilities
        V = [X * β[:, j] + Z[:, j] for j in 1:J]
        
        # Compute probabilities
        P = exp.(hcat(V...))
        P = P ./ sum(P, dims=2)
        
        return P
    end

    # Log-likelihood function
    function log_likelihood(θ, X, Z, y)
        J = size(Z, 2)
        β = reshape(θ[1:end-1], :, J-1)
        γ = θ[end]
        
        P = mnl_probs(β, X, Z .- Z[:, J])
        
        # Add column for base alternative
        P = hcat(P, 1 .- sum(P, dims=2))
        
        # Compute log-likelihood
        ll = sum(log.(P[CartesianIndex.(1:size(P,1), y)]))
        
        return -ll  # Negative because we're minimizing
    end

    # Gradient function using ForwardDiff
    function gradient(θ, X, Z, y)
        return ForwardDiff.gradient(θ -> log_likelihood(θ, X, Z, y), θ)
    end

    # Initial values (you might want to use estimates from PS3 as suggested)
    β_init = zeros(size(X, 2), size(Z, 2) - 1)
    γ_init = 0.0
    θ_init = [vec(β_init); γ_init]

    # Optimization
    result = optimize(θ -> log_likelihood(θ, X, Z, y), 
                      θ -> gradient(θ, X, Z, y),
                      θ_init, 
                      BFGS(), 
                      Optim.Options(show_trace = true, iterations = 1000))

    # Extract results
    θ_hat = Optim.minimizer(result)
    β_hat = reshape(θ_hat[1:end-1], :, size(Z, 2) - 1)
    γ_hat = θ_hat[end]

    # Compute standard errors
    H = ForwardDiff.hessian(θ -> log_likelihood(θ, X, Z, y), θ_hat)
    se = sqrt.(diag(inv(H)))

    # Print results
    println("Multinomial Logit Results:")
    println("Estimated coefficients (β):")
    display(β_hat)
    println("\nEstimated γ:")
    println(γ_hat)
    println("\nStandard errors:")
    display(se)

    # Function for Gauss-Legendre quadrature
    function lgwt(N, a, b)
        N, a, b = promote(N, a, b)
        x = zeros(N)
        w = zeros(N)
        
        m = (N + 1) ÷ 2
        xm = 0.5 * (b + a)
        xl = 0.5 * (b - a)
        
        for i = 1:m
            z = cos(π * (i - 0.25) / (N + 0.5))
            
            while true
                p1 = 1.0
                p2 = 0.0
                for j = 1:N
                    p3 = p2
                    p2 = p1
                    p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
                end
                pp = N * (z * p1 - p2) / (z * z - 1)
                z1 = z
                z = z1 - p1 / pp
                if abs(z - z1) < eps(z)
                    break
                end
            end
            
            x[i] = xm - xl * z
            x[N + 1 - i] = xm + xl * z
            w[i] = 2 * xl / ((1 - z * z) * pp * pp)
            w[N + 1 - i] = w[i]
        end
        
        return x, w
    end

    # Practice with quadrature
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)

    println("\nQuadrature Results:")
    println("Integral of density:")
    println(sum(weights .* pdf.(d, nodes)))

    println("Expectation:")
    println(sum(weights .* nodes .* pdf.(d, nodes)))

    # Quadrature for N(0, 2) with 7 and 10 points
    d2 = Normal(0, 2)
    nodes7, weights7 = lgwt(7, -5*2, 5*2)
    nodes10, weights10 = lgwt(10, -5*2, 5*2)

    println("\nVariance with 7 points:")
    println(sum(weights7 .* nodes7.^2 .* pdf.(d2, nodes7)))

    println("Variance with 10 points:")
    println(sum(weights10 .* nodes10.^2 .* pdf.(d2, nodes10)))

    # Monte Carlo integration
    function monte_carlo_integral(f, a, b, D)
        X = rand(Uniform(a, b), D)
        return (b - a) * mean(f.(X))
    end

    d2 = Normal(0, 2)
    a, b = -5*2, 5*2

    println("\nMonte Carlo Integration Results:")
    println("Variance (D=1,000,000):")
    println(monte_carlo_integral(x -> x^2 * pdf(d2, x), a, b, 1_000_000))

    println("Mean (D=1,000,000):")
    println(monte_carlo_integral(x -> x * pdf(d2, x), a, b, 1_000_000))

    println("Integral of density (D=1,000,000):")
    println(monte_carlo_integral(x -> pdf(d2, x), a, b, 1_000_000))

    println("Variance (D=1,000):")
    println(monte_carlo_integral(x -> x^2 * pdf(d2, x), a, b, 1_000))

    # Mixed logit (not run, just structure)
    function mixed_logit_ll(θ, X, Z, y, nodes, weights)
        # Extract parameters
        β = reshape(θ[1:end-2], :, size(Z, 2) - 1)
        μ_γ, σ_γ = θ[end-1:end]
        
        # Compute probabilities for each quadrature point
        function compute_probs(γ)
            V = [X * β[:, j] .+ γ * (Z[:, j] .- Z[:, end]) for j in 1:size(Z, 2)-1]
            P = exp.(hcat(V...))
            P = hcat(P, ones(size(P, 1)))
            P = P ./ sum(P, dims=2)
            return P
        end
        
        # Compute likelihood
        ll = 0.0
        for (node, weight) in zip(nodes, weights)
            γ = μ_γ + σ_γ * node
            P = compute_probs(γ)
            ll += weight * sum(log.(P[CartesianIndex.(1:size(P,1), y)]))
        end
        
        return -ll  # Negative because we're minimizing
    end

    println("\nMixed Logit Model:")
    println("(Not run due to long computation time)")
    println("To run the mixed logit model, uncomment the following code:")
    println("nodes, weights = lgwt(7, -4, 4)")
    println("θ_init_mixed = [vec(β_hat); γ_hat; 1.0]  # Initial σ_γ set to 1.0")
    println("result_mixed = optimize(θ -> mixed_logit_ll(θ, X, Z, y, nodes, weights), ")
    println("                        θ_init_mixed, ")
    println("                        BFGS(), ")
    println("                        Optim.Options(show_trace = true, iterations = 1000))")
    println("θ_hat_mixed = Optim.minimizer(result_mixed)")
    println("β_hat_mixed = reshape(θ_hat_mixed[1:end-2], :, size(Z, 2) - 1)")
    println("μ_γ_hat, σ_γ_hat = θ_hat_mixed[end-1:end]")
    println("println(\"Mixed Logit Results:\")")
    println("println(\"β_hat:\")")
    println("display(β_hat_mixed)")
    println("println(\"μ_γ_hat:\")")
    println("println(μ_γ_hat)")
    println("println(\"σ_γ_hat:\")")
    println("println(σ_γ_hat)")
end

# Call the function 
run_econometrics_analysis()


#:::::::::::::::::::::::::::::
# Question 7
#:::::::::::::::::::::::::::::

# Unit test
using Test

@testset "Multinomial Logit Tests" begin
    # Test data
    X_test = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    Z_test = [0.5 1.0 1.5; 1.0 1.5 2.0; 1.5 2.0 2.5]
    y_test = [1, 2, 3]
    β_test = [0.5 1.0; 1.0 1.5]
    γ_test = 1.0

    # Test mnl_probs function
    P_test = mnl_probs(β_test, X_test, Z_test)
    @test size(P_test) == (3, 3)
    @test all(sum(P_test, dims=2) .≈ 1)

    # Test log_likelihood function
    θ_test = [vec(β_test); γ_test]
    ll_test = log_likelihood(θ_test, X_test, Z_test, y_test)
    @test typeof(ll_test) == Float64
    @test ll_test < 0  # Log-likelihood should be negative

    # Test gradient function
    grad_test = gradient(θ_test, X_test, Z_test, y_test)
    @test length(grad_test) == length(θ_test)
end

@testset "Quadrature Tests" begin
    # Test lgwt function
    nodes, weights = lgwt(5, -1, 1)
    @test length(nodes) == 5
    @test length(weights) == 5
    @test sum(weights) ≈ 2  # Integral of 1 over [-1, 1]

    # Test quadrature approximation
    d = Normal(0, 1)
    @test sum(weights .* pdf.(d, nodes)) ≈ 1 atol=1e-6
    @test sum(weights .* nodes .* pdf.(d, nodes)) ≈ 0 atol=1e-6
end

@testset "Monte Carlo Integration Tests" begin
    # Test monte_carlo_integral function
    f(x) = x^2
    integral = monte_carlo_integral(f, 0, 1, 1_000_000)
    @test integral ≈ 1/3 atol=1e-3
end

@testset "Mixed Logit Tests" begin
    # Test data
    X_test = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    Z_test = [0.5 1.0 1.5; 1.0 1.5 2.0; 1.5 2.0 2.5]
    y_test = [1, 2, 3]
    β_test = [0.5 1.0; 1.0 1.5]
    μ_γ_test = 1.0
    σ_γ_test = 0.5
    θ_test = [vec(β_test); μ_γ_test; σ_γ_test]

    # Test mixed_logit_ll function
    nodes, weights = lgwt(7, -4, 4)
    ll_test = mixed_logit_ll(θ_test, X_test, Z_test, y_test, nodes, weights)
    @test typeof(ll_test) == Float64
    @test ll_test < 0  # Log-likelihood should be negative
end