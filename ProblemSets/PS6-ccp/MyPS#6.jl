

import Pkg
Pkg.add("StatsModels")
using DataFrames, CSV, HTTP, DataFramesMeta, GLM, LinearAlgebra, Random, Statistics, Optim, StatsModels

# Function to load and preprocess data
function load_and_preprocess_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df,1))
    return df
end

# Function to reshape data to long format
function reshape_data(df)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, 
                  :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                  :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, 
                  :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, 
                  :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long, [:bus_id,:time])
    return df_long
end

# Function to estimate flexible logit model
function estimate_flexible_logit(df_long)
    df_long.Mileage2 = df_long.Odometer.^2
    df_long.RouteUsage2 = df_long.RouteUsage.^2
    df_long.Time = df_long.time
    df_long.Time2 = df_long.time.^2

    formula = @formula(Y ~ (Odometer + Mileage2 + RouteUsage + RouteUsage2 + Branded + Time + Time2)^7)
    model = glm(formula, df_long, Binomial(), LogitLink())
    
    return model
end

# Function to create state transition matrices
function create_grids()
    zval = collect(0.25:0.01:1.25)
    zbin = length(zval)
    xval = collect(0:0.125:6)
    xbin = length(xval)
    xtran = zeros(zbin*xbin, xbin)
    
    for x = 1:xbin
        for z = 1:zbin
            for xp = x:xbin
                xtran[(z-1)*xbin+x, xp] = exp(-zval[z]*(xval[xp]-xval[x])) - exp(-zval[z]*(xval[xp]+0.125-xval[x]))
            end
            xtran[(z-1)*xbin+x, :] = xtran[(z-1)*xbin+x, :] ./ sum(xtran[(z-1)*xbin+x, :])
        end
    end
    
    return zval, zbin, xval, xbin, xtran
end

# Function to compute future value terms
function compute_future_values(flexible_logit_model, zval, xval, zbin, xbin, xtran, β)
    # Create state space for prediction
    state_space = DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded = zeros(zbin * xbin),
        time = zeros(zbin * xbin)
    )
    state_space.Mileage2 = state_space.Odometer.^2
    state_space.RouteUsage2 = state_space.RouteUsage.^2
    state_space.Time2 = state_space.time.^2

    # Initialize future value array
    FV = zeros(zbin * xbin, 2, 21)  # 21 because we have T=20 and need T+1

    # Backwards recursion
    for t in 20:-1:1
        for b in 0:1
            state_space.time .= t
            state_space.Branded .= b
            state_space.Time .= t
            state_space.Time2 .= t^2

            p0 = predict(flexible_logit_model, state_space)
            FV[:, b+1, t] = -β .* log.(p0)
        end
    end

    return FV, xval, zval
end

# Function to estimate structural parameters

function estimate_structural_parameters(df_long, FV, xtran, xbin, β, xval, zval)
    # Map future values to original data
    fvt1 = zeros(size(df_long, 1))
    for i in 1:size(df_long, 1)
        t = Int(df_long.time[i])
        b = Int(df_long.Branded[i])
        
        x_index = findfirst(x -> x >= df_long.Odometer[i], xval)
        z_index = findfirst(x -> x >= df_long.RouteUsage[i], zval)
        
        x = isnothing(x_index) ? xbin : x_index
        z = isnothing(z_index) ? length(zval) : z_index
        
        row0 = (z-1)*xbin + 1
        row1 = x + (z-1)*xbin
        
        fvt1[i] = (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, b+1, t+1]
    end

    # Add future value to dataframe
    df_long.fv = fvt1

    # Estimate structural parameters
    model = glm(@formula(Y ~ Odometer + Branded + fv), df_long, Binomial(), LogitLink())

    return coef(model)
end

# Main function
function main()
    # Question 1: Load and reshape data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = load_and_preprocess_data(url)
    df_long = reshape_data(df)
    println("Data loaded and reshaped.")

    # Question 2: Estimate flexible logit model
    flexible_logit_model = estimate_flexible_logit(df_long)
    println("Flexible logit model estimated:")
    println(flexible_logit_model)

    # Question 3: Dynamic estimation with CCPs
    # 3a: Construct state transition matrices
    zval, zbin, xval, xbin, xtran = create_grids()
    println("State transition matrices constructed.")

    # 3b: Compute future value terms
    β = 0.9  # Given discount factor
    FV, xval, zval = compute_future_values(flexible_logit_model, zval, xval, zbin, xbin, xtran, β)
    println("Future value terms computed.")

    # 3c: Estimate structural parameters
    θ_hat = estimate_structural_parameters(df_long, FV, xtran, xbin, β, xval, zval)
    println("Structural parameters estimated:")
    println("θ_0 = ", θ_hat[1])
    println("θ_1 = ", θ_hat[2])
    println("θ_2 = ", θ_hat[3])

    # 3d-3f: These steps are implemented within the functions above

    # 3g: Wrap everything in a function (already done in this main() function)

    # 3h: Time the execution
    println("Timing the entire estimation process:")
    @time main()
end

# Run the main function
main()

# Question 4: Unit tests
using Test

@testset "Bus Engine Replacement Model Tests" begin
    @test size(reshape_data(load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")), 1) > 0
    
    @test length(create_grids()) == 5
    
    zval, zbin, xval, xbin, xtran = create_grids()
    df = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    df_long = reshape_data(df)
    flexible_logit_model = estimate_flexible_logit(df_long)
    
    FV, _, _ = compute_future_values(flexible_logit_model, zval, xval, zbin, xbin, xtran, 0.9)
    @test size(FV, 3) == 21
    
    @test length(estimate_structural_parameters(df_long, FV, xtran, xbin, 0.9, xval, zval)) == 3
end