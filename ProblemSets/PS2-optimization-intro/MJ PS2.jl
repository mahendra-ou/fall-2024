# ECON 6343: ECONOMETRICS II PS#2 (FALL 24)
# NAME: Mahendra Jamankar

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

using Pkg
Pkg.add(["Optim", "HTTP", "GLM", "LinearAlgebra", "Random", "Statistics", "DataFrames", "CSV", "FreqTables"])
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTable


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 1: Basic optimization in Julia
using Optim

f(x) = -x^4 - 10x^3 - 2x^2 - 3x - 2
neg_f(x) = -f(x)

result = optimize(neg_f, 0.0, LBFGS())

println("Optimal x: ", Optim.minimizer(result))
println("Maximum f(x): ", f(Optim.minimizer(result)))

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question2: Compute OlS estimate of linear regression

using DataFrames, CSV, HTTP
# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]  #define x and y
y = df.married .== 1

# Objective function: sum of squared residuals
ols_obj(β) = sum((y - X * β)^2)  #

result = optimize(ols_obj, zeros(size(X, 2)), LBFGS())  #optimize using LBFGS
println("Estimated coefficients (β): ", Optim.minimizer(result))  #display the results

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 3: Estimate Logit Likelihood function

# Define the negative log-likelihood function
function neg_log_likelihood(β)
    Xβ = X * β
    ll = sum(y .* Xβ .- log1p.(exp.(Xβ)))  # log1p(exp(x)) computes log(1 + exp(x))
    return -ll
end

result = optimize(neg_log_likelihood, zeros(size(X, 2)), LBFGS())  #optimize using LBFGS
println("Estimated coefficients (β): ", Optim.minimizer(result))   #display the results

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 4: Use glm() function to check answer
# Prepare the data
df.race_binary = df.race .== 1
df.collgrad_binary = df.collgrad .== 1

# Fit the logistic regression model using GLM
model = glm(@formula(married ~ age + race_binary + collgrad_binary), df, Binomial())

# Display the results
println("GLM Estimated coefficients (β):")
println(coef(model))

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 5: Estimate multinomial logit model

df = dropmissing(df, :occupation)  # Drop missing values

# Aggregate occupation categories (example aggregation)
df.occupation = recode(df.occupation, 
                       "1" => "Professional",
                       "2" => "Technical",
                       "3" => "Clerical",
                       "4" => "Sales",
                       "5" => "Service",
                       "6" => "Laborer",
                       "7" => "Other")

# Convert occupation to categorical variable
df.occupation = categorical(df.occupation)

# Re-define x and y objects
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.occupation

# Define the negative log-likelihood function for multinomial logit
function neg_log_likelihood(β)
    K = size(X, 2)
    J = length(levels(y)) - 1  # Number of alternatives minus 1
    β_matrix = reshape(β, K, J)  # Transform β into a K × J matrix
    
    ll = 0.0
    for i in 1:size(X, 1)
        _ = X[i, :]
        y_i = y[i]
        β_j = β_matrix[:, y_i]  # β for the observed category
        exp_term = exp.(X[i, :] * β_matrix)
        ll += X[i, :] * β_j - log(sum(exp_term))
    end
    
    return -ll
end

# Starting values
n_params = size(X, 2) * (length(levels(y)) - 1)
starting_values = randn(n_params)  # or use zeros or another initialization

# Optimize using LBFGS with a higher tolerance for convergence
result = optimize(neg_log_likelihood, starting_values, LBFGS(g_tol=1e-5))

# Display results
β_estimates = Optim.minimizer(result)
println("Estimated coefficients (β):")
println(reshape(β_estimates, size(X, 2), length(levels(y)) - 1))

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 6: Wrap up code

using DataFrames, CSV, HTTP, Optim, GLM, LinearAlgebra

function basic_optimization()
    # Define the function f(x) = −x^4 −10*x^3 −2x^2 −3x−2
    function f(x)
        return -x^4 - 10*x^3 - 2*x^2 - 3*x - 2
    end

    # Define the negative of the function for minimization
    function neg_f(x)
        return -f(x)
    end

    # Optimize using LBFGS
    result = optimize(neg_f, [0.0], LBFGS(g_tol=1e-5))
    println("Basic optimization result (x): ", Optim.minimizer(result))
end

function ols_estimation()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Prepare X and y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    # Define the OLS objective function
    function ols_obj(β)
        residuals = y .- X * β
        return sum(residuals .^ 2)
    end

    # Optimize using LBFGS
    result = optimize(ols_obj, zeros(size(X, 2)), LBFGS(g_tol=1e-5))
    println("OLS estimates (β): ", Optim.minimizer(result))
end

function logit_estimation()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Prepare X and y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    # Define the negative log-likelihood function for logistic regression
    function neg_log_likelihood(β)
        Xβ = X * β
        ll = sum(y .* Xβ .- log1p.(exp.(Xβ)))  # log1p(exp(x)) computes log(1 + exp(x))
        return -ll
    end

    # Optimize using LBFGS
    result = optimize(neg_log_likelihood, zeros(size(X, 2)), LBFGS(g_tol=1e-5))
    println("Logit model estimated coefficients (β): ", Optim.minimizer(result))

    # Fit the logistic regression model using GLM
    model = glm(@formula(married ~ age + race + collgrad), df, Binomial())
    println("GLM estimated coefficients (β):")
    println(coef(model))
end

function multinomial_logit_estimation()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Clean data: Remove rows with missing occupation
    df = dropmissing(df, :occupation)

    # Aggregate occupation categories
    df.occupation = recode(df.occupation, 
                           "1" => "Professional",
                           "2" => "Technical",
                           "3" => "Clerical",
                           "4" => "Sales",
                           "5" => "Service",
                           "6" => "Laborer",
                           "7" => "Other")

    # Convert occupation to categorical variable
    df.occupation = categorical(df.occupation)

    # Create indicator variables for occupation categories
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    # Define the negative log-likelihood function for multinomial logit
    function neg_log_likelihood(β)
        K = size(X, 2)
        J = length(levels(y)) - 1  # Number of alternatives minus 1
        β_matrix = reshape(β, K, J)  # Transform β into a K × J matrix

        ll = 0.0
        for i in 1:size(X, 1)
            x_i = X[i, :]
            y_i = y[i]
            β_j = β_matrix[:, y_i]  # β for the observed category
            exp_term = exp.(X[i, :] * β_matrix)
            ll += X[i, :] * β_j - log(sum(exp_term))
        end

        return -ll
    end

    # Starting values
    n_params = size(X, 2) * (length(levels(y)) - 1)
    starting_values = randn(n_params)  # or use zeros or another initialization

    # Optimize using LBFGS with a higher tolerance for convergence
    result = optimize(neg_log_likelihood, starting_values, LBFGS(g_tol=1e-5))
    β_estimates = Optim.minimizer(result)

    # Display results
    println("Multinomial logit model estimated coefficients (β):")
    println(reshape(β_estimates, size(X, 2), length(levels(y)) - 1))
end

# Call each function to perform the respective estimations
println("Basic Optimization Results:")
basic_optimization()

println("\nOLS Estimation Results:")
ols_estimation()

println("\nLogit Model Estimation Results:")
logit_estimation()

println("\nMultinomial Logit Model Estimation Results:")
multinomial_logit_estimation()

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Question 7: AI write unit test
using Test

function run_tests()
    # Dummy data for testing
    X_test = [1.0 25.0 1.0 0.0; 1.0 30.0 0.0 1.0]
    y_test = [1.0; 0.0]
    beta_test = [0.5, -0.1, 0.05, 0.3]

    # Expected SSR (sum of squared residuals)
    expected_ssr = (y_test .- X_test * beta_test)' * (y_test .- X_test * beta_test)

    # Test OLS function
    @testset "OLS Function Test" begin
        @test ols(beta_test, X_test, y_test) ≈ expected_ssr atol=1e-6
    end

    # Dummy data for logit test
    beta_logit_test = [0.5, -0.1, 0.05, 0.3]
    X_logit_test = [1.0 25.0 1.0 0.0; 1.0 30.0 0.0 1.0]
    y_logit_test = [1.0; 0.0]

    # Expected log likelihood
    p_test = 1.0 ./ (1.0 .+ exp.(-X_logit_test * beta_logit_test))
    expected_loglik = -sum(y_logit_test .* log.(p_test) .+ (1 .- y_logit_test) .* log.(1 .- p_test))

    # Test Logit function
    @testset "Logit Function Test" begin
        @test logit_loglik(beta_logit_test, X_logit_test, y_logit_test) ≈ expected_loglik atol=1e-6
    end
end

# Run all tests
run_tests()



