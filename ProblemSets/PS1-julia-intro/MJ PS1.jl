# Econ 6343 - Econometrics PS-1
using Pkg
Pkg.add("JLD")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("FreqTables")
Pkg.add("Distributions")

using CSV
using DataFrames
using Distributions
using FreqTables
using JLD
using LinearAlgebra
using Random
using Statistics


# 0. gitHub setup 
# 1. Initializing variables and practice with basic matrix operations. 
# (a) Create four matrices of random numbers. 
Random.seed!(1234)
A = rand(Uniform(-5, 10), 10, 7)
B = rand(Normal(-2, 15), 10, 7)
C = [A[1:5, 1:5] B[1:5, 6:7]]
D = [A[i, j] ≤ 0 ? A[i, j] : 0 for i in 1:10, j in 1:7]
# Print the matrices to verify
println("Matrix A:\n", A)
println("Matrix B:\n", B)
println("Matrix C:\n", C)
println("Matrix D:\n", D)

# (b) Use built-in-Julia to number of element of A 
num_elements_A = length(A)
println("Number of elements in matrix A: ", num_elements_A)
# (c) Unique elements of D
num_unique_elements_D = length(unique(vec(D)))
println("Number of unique elements in matrix D: ", num_unique_elements_D)
# (d) Create new matrix E 
E = reshape(vec(B), 10, 7)
println("Matrix E:\n", E)
# (e) Create a new array F. 
F = cat(A, B, dims=3)
#(f) Use the permutedims function to twist F 
F_permuted = permutedims(F, (2,1,3))
# (g) Create matrix G 
G = kronecker(B,C)
println("kronecker product of B and C: ", G)
# Attempt to compute C⊗F
try
    kron(C,F)
catch e
    println("Error: ", e)
end

# (h) save all matrices file named matrixpractice
save("matrixpractice.jld", "A" => A, "B" => B, "C" => C, "D" => D, "E" => E, "F" => F, "G" => G)
# (i) Save only the matrices ABCD
save("firstmatrix.jld", "A" => A, "B" => B, "C" => C, "D" => D)
# (j) Export C as a csv file called Cmatrix 
C_df = DataFrame(C)
CSV.write("Cmatrix.csv", C_df)
# (k) Export D as a tab-delimited .dat file called Dmatrix
D_df = DataFrame(D)
CSV.write("Dmatrix.dat", D_df, delim='\t')

#(i) Wrap a function definition around all of the code for q.1. 
function q1()
    Random.seed!(1234)
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = A .* (A .≤ 0)
    E = reshape(B, :)
    F = cat(A, B, dims=3)
    F_permuted = permutedims(F, (2,1,3))
    G = kron(B, C)
    save("matrixpractice.jld", "A" => A, "B" => B, "C" => C, "D" => D, "E" => E, "F" => F, "G" => G)
    save("firstmatrix.jld", "A" => A, "B" => B, "C" => C, "D" => D)
    CSV.write("Cmatrix.csv", DataFrame(C))
    CSV.write("Dmatrix.dat", DataFrame(D), delim='\t')
    return A, B, C, D
end
A, B, C, D = q1()


# 2. Practice with loops and comprehension 

function q2(A, B, C)
    # (a) Element-wise product of A and B
    AB = [A[i,j] * B[i,j] for i in 1:size(A,1), j in 1:size(A,2)]
    AB2 = A .* B
    
    # (b) Extract elements from C between -5 and 5
    Cprime = [C[i,j] for i in 1:size(C,1), j in 1:size(C,2) if -5 <= C[i,j] <= 5]
    Cprime2 = C[(C .>= -5) .& (C .<= 5)]
    
    # (c) Create 3D array X
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    for t in 1:T
        X[:,1,t] = 1.0
        X[:,2,t] = rand(Binomial(1, 0.75 * (6 - t) / 5), N)
        X[:,3,t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        X[:,4,t] = rand(Normal(pi * (6 - t) / 3, 1 / exp(1)), N)
        X[:,5,t] = rand(Binomial(20, 0.6), N)
        X[:,6,t] = rand(Binomial(20, 0.5), N)
    end
    
    # (d) Create matrix β
    β = [1.0 + 0.25*(t-1), log(t), -sqrt(t), exp(t) - exp(t+1), t, t/3.0 for t in 1:T]
    
    # (e) Create matrix Y = X * β + ε
    Y = Array{Float64}(undef, N, T)
    for t in 1:T
        ε = rand(Normal(0, 0.36), N)
        Y[:,t] = X[:,:,t] * β[:,t] + ε
    end
end

#(f) wrap a function definition
q2(A, B, C)

# 3. Reading in Data and calculating summary statistics 

function q3()
    # (a) Import nlsw88.csv
    df = CSV.read("nlsw88.csv", DataFrame; missingstring="NA")
    CSV.write("nlsw88_processed.csv", df)
    
    # (b) Percentage never married and college graduates
    percent_never_married = sum(df.marital_status .== "Never married") / nrow(df) * 100
    percent_college_graduates = sum(df.education .== "College graduate") / nrow(df) * 100
    
    # (c) Frequency table for race
    race_freq_table = freqtable(df, :race)
    
    # (d) Summary statistics
    summarystats = describe(df)
    missing_grades = sum(ismissing.(df.grade))
    
    # (e) Cross-tabulation of industry and occupation
    industry_occupation_crosstab = freqtable(df, :industry, :occupation)
    
    # (f) Mean wage by industry and occupation
    df_subset = select(df, :industry, :occupation, :wage)
    mean_wage = combine(groupby(df_subset, [:industry, :occupation]), :wage => mean)
end

#(g) wrap a function definition. 
q3()

# 4. Practice with functions 

function q4()
    # (a) Load firstmatrix.jld
    firstmatrix = JLD.load("firstmatrix.jld")
    A = firstmatrix["A"]
    B = firstmatrix["B"]
    
    # (b) Define matrixops function
    function matrixops(A, B)
        if size(A) != size(B)
            error("inputs must have the same size")
        end
        
        # (c) Element-by-element product, A'B, sum of A+B
        elementwise_product = A .* B
        matrix_product = A' * B
        sum_elements = sum(A + B)
        
        return elementwise_product, matrix_product, sum_elements
    end
    
    # (d) Evaluate matrixops with A and B
    elementwise_product, matrix_product, sum_elements = matrixops(A, B)
    
    # (e) Evaluate matrixops with C and D
    elementwise_product_CD, matrix_product_CD, sum_elements_CD = matrixops(C, D)
    
    # (f) Evaluate matrixops with columns from nlsw88
    df = CSV.read("nlsw88_processed.csv", DataFrame; missingstring="NA")
    ttl_exp = convert(Array, df.ttl_exp)
    wage = convert(Array, df.wage)
    elementwise_product_exp_wage, matrix_product_exp_wage, sum_elements_exp_wage = matrixops(ttl_exp, wage)
end

#(h) wrap a function definition
q4()

# 5. Write unit tests for each of the functions you’ve created and run them to verify that they work as expected.

function test_q1()
    A, B, C, D = q1()
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
end

function test_q2(A, B, C)
    q2(A, B, C)
    # Add specific tests for outputs
end

function test_q3()
    q3()
    # Add specific tests for outputs
end

function test_q4()
    q4()
    # Add specific tests for outputs
end

@testset "Problem Set 1 Tests" begin
    test_q1()
    test_q2(A, B, C)
    test_q3()
    test_q4()
end
