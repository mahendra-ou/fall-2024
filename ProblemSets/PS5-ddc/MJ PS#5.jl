# problem set 4 solutions
# Name: Mahendra Jamankar

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM


# read in function to create state transitions for dynamic model
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# load in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# create bus id variable
df = @transform(df, :bus_id = 1:size(df,1))

# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
static_model = glm(@formula(decision ~ mileage + brand), df_long, Binomial(), LogitLink())
θ_static = coef(static_model)
println("Static model coefficients:")
println("θ₀ (Intercept): ", θ_static[1])
println("θ₁ (Mileage): ", θ_static[2])
println("θ₂ (Brand): ", θ_static[3])

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Load in the data 
url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df_dynamic = CSV.read(HTTP.get(url_dynamic).body, DataFrame)

# Convert relevant columns to arrays
Y = Matrix(df_dynamic[:, r"^Y\d+$"])
Odo = Matrix(df_dynamic[:, r"^Odo\d+$"])
Xst = Matrix(df_dynamic[:, r"^Xst\d+$"])
Zst = Matrix(df_dynamic[:, r"^Zst\d+$"])
B = df_dynamic.brand

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function create_grids()
    zval = 0.25:0.01:1.25
    xval = 0:0.125:20
    zbin, xbin = length(zval), length(xval)
    xtran = zeros(zbin * xbin, xbin)
    
    for (z, x₂) in enumerate(zval), (x, x₁) in enumerate(xval)
        row = x + (z-1)*xbin
        xtran[row, x:end] = exp.(-x₂ * (xval[x:end] .- x₁)) .- exp.(-x₂ * (xval[x:end] .+ 0.125 .- x₁))
        xtran[row, :] ./= sum(xtran[row, :])
    end
    
    return zval, zbin, xval, xbin, xtran
end

zval, zbin, xval, xbin, xtran = create_grids()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3c: Compute the future value terms 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function compute_future_values(θ, β, T, zbin, xbin, xval, xtran)
    FV = zeros(zbin * xbin, 2, T + 1)
    for t in T:-1:1, b in 0:1, z in 1:zbin
        rows = (z-1)*xbin+1:z*xbin
        v1 = @. θ[1] + θ[2] * xval + θ[3] * b + β * (xtran[rows,:]' * FV[rows,b+1,t+1])
        v0 = β * (xtran[rows[1],:]' * FV[rows,b+1,t+1])
        FV[rows,b+1,t] .= β * log.(exp.(v0) + exp.(v1))
    end
    return FV
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3d: generate the likelihood function
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function log_likelihood(θ, β, T, xbin, xtran, Y, Odo, Xst, Zst, B, FV)
    ll = 0.0
    for i in 1:size(Y, 1), t in 1:T
        row0, row1 = 1 + (Zst[i,t] - 1) * xbin, Xst[i,t] + (Zst[i,t] - 1) * xbin
        v1t_v0t = θ[1] + θ[2] * Odo[i,t] + θ[3] * B[i] + 
                  β * ((xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1])
        p1t = 1 / (1 + exp(-v1t_v0t))
        ll += Y[i,t] * log(p1t) + (1 - Y[i,t]) * log(1 - p1t)
    end
    return ll
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3e: Wrap all code from c and d
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@views @inbounds function objective(θ, β, T, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B)
    FV = compute_future_values(θ, β, T, zbin, xbin, xval, xtran, B)
    return -log_likelihood(θ, β, T, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B, FV)
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3f:
#:::::::::::::::::::::::::::::::::::::::::::::::::::

@views @inbounds function objective(θ, β, T, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B)
    # Function body...
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3g: Wrap all code 
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function estimate_dynamic_model()
    # All the code goes here
    
    # Optimization
    β = 0.9
    T = 20
    initial_θ = θ_static  # Use static model estimates as initial values
    
    result = optimize(θ -> objective(θ, β, T, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B),
                      initial_θ, BFGS(), Optim.Options(show_trace = true))
    
    θ_dynamic = Optim.minimizer(result)
    
    println("Dynamic model coefficients:")
    println("θ₀ (Intercept): ", θ_dynamic[1])
    println("θ₁ (Mileage): ", θ_dynamic[2])
    println("θ₂ (Brand): ", θ_dynamic[3])
    
    return θ_dynamic
end

# Run the estimation
θ_dynamic = estimate_dynamic_model()


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: AI unit test
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Test

@testset "Dynamic Bus Engine Replacement Model Tests" begin
    @testset "create_grids() function" begin
        zval, zbin, xval, xbin, xtran = create_grids()
        @test length(zval) == zbin
        @test length(xval) == xbin
        @test size(xtran) == (zbin * xbin, xbin)
        @test all(sum(xtran, dims=2) .≈ 1)  # Each row should sum to 1
    end

    @testset "compute_future_values() function" begin
        # Create dummy inputs
        θ_test = [1.0, -0.1, 0.5]
        β_test = 0.9
        T_test = 5
        zbin_test, xbin_test = 3, 4
        xval_test = [0.0, 0.5, 1.0, 1.5]
        xtran_test = rand(zbin_test * xbin_test, xbin_test)
        xtran_test = xtran_test ./ sum(xtran_test, dims=2)  # Normalize rows
        B_test = [0, 1]

        FV = compute_future_values(θ_test, β_test, T_test, zbin_test, xbin_test, xval_test, xtran_test, B_test)
        
        @test size(FV) == (zbin_test * xbin_test, 2, T_test + 1)
        @test all(FV[:, :, end] .== 0)  # Last period should be all zeros
        @test all(FV .>= 0)  # All future values should be non-negative
    end

    @testset "log_likelihood() function" begin
        # Create dummy inputs
        θ_test = [1.0, -0.1, 0.5]
        β_test = 0.9
        T_test = 5
        zbin_test, xbin_test = 3, 4
        xval_test = [0.0, 0.5, 1.0, 1.5]
        xtran_test = rand(zbin_test * xbin_test, xbin_test)
        xtran_test = xtran_test ./ sum(xtran_test, dims=2)  # Normalize rows
        N_test = 10
        Y_test = rand(0:1, N_test, T_test)
        Odo_test = rand(xval_test, N_test, T_test)
        Xst_test = rand(1:xbin_test, N_test, T_test)
        Zst_test = rand(1:zbin_test, N_test, T_test)
        B_test = rand(0:1, N_test)

        FV_test = compute_future_values(θ_test, β_test, T_test, zbin_test, xbin_test, xval_test, xtran_test, B_test)
        ll = log_likelihood(θ_test, β_test, T_test, zbin_test, xbin_test, xval_test, xtran_test, Y_test, Odo_test, Xst_test, Zst_test, B_test, FV_test)

        @test typeof(ll) == Float64
        @test !isnan(ll) && !isinf(ll)
    end

    @testset "objective() function" begin
        # Create dummy inputs (similar to log_likelihood test)
        θ_test = [1.0, -0.1, 0.5]
        β_test = 0.9
        T_test = 5
        zbin_test, xbin_test = 3, 4
        xval_test = [0.0, 0.5, 1.0, 1.5]
        xtran_test = rand(zbin_test * xbin_test, xbin_test)
        xtran_test = xtran_test ./ sum(xtran_test, dims=2)  # Normalize rows
        N_test = 10
        Y_test = rand(0:1, N_test, T_test)
        Odo_test = rand(xval_test, N_test, T_test)
        Xst_test = rand(1:xbin_test, N_test, T_test)
        Zst_test = rand(1:zbin_test, N_test, T_test)
        B_test = rand(0:1, N_test)

        obj_value = objective(θ_test, β_test, T_test, zbin_test, xbin_test, xval_test, xtran_test, Y_test, Odo_test, Xst_test, Zst_test, B_test)

        @test typeof(obj_value) == Float64
        @test !isnan(obj_value) && !isinf(obj_value)
        @test obj_value > 0  # The negative log-likelihood should be positive
    end
end


