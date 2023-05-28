# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
# 
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using LinearAlgebra, Statistics, Distributions, TimeSeries

#-------------------------------------------------------------------------------
# struct FactorModel
#-------------------------------------------------------------------------------
struct FactorModel
    """
    FactorModel structure and constructor 
        Use a linear regression model to estimate the asset means and covariance 
        matrix. If provided, use a hidden Markov model to calibrate a linear 
        regression model per state.

    Inputs.
        frets: T x M Timeseries of feature returns for M features and T observations.
        arets: T x N Timeseries of asset returns for N assets and T observations
        lookback: Number of observations to use to conduct the regression
        rsmodel (Optional): Object containing a T x S timeseries of smoothed 
            probabilities for S states and T observations, as well as the transition 
            probabilities conditional on the current estimated state

    Outputs. FactorModel struct with the following fields
        mu: N x 1 vector of intercepts 
        sigma: N x N covariance matrix   
    """
    mu::Vector{Float64}
    sigma::Symmetric{Float64,Matrix{Float64}}

    function FactorModel(frets::TimeArray, 
        arets::TimeArray, 
        lookback::Int64)

        mu, sigma = fmodel(frets, arets, lookback)
        new(mu, sigma)
    end

    function FactorModel(frets::TimeArray, 
        arets::TimeArray, 
        lookback::Int64,
        rsmodel::HMM)

        mu, sigma = fmodel_rs(frets, arets, lookback, rsmodel)
        new(mu, sigma)
    end
end

#-------------------------------------------------------------------------------
# Function fmodel:
#-------------------------------------------------------------------------------
function fmodel(frets::TimeArray, arets::TimeArray, lookback::Int64)
    """
    Construct a factor model by performing a linear regression.
        The factor model is used to estimate the asset means and covariance matrix

    Inputs
        frets: T x M Timeseries of feature returns for M features and T observations.
        arets: T x N Timeseries of asset returns for N assets and T observations
        lookback: Number of observations to use to conduct the regression

    Outputs
        mu: N x 1 vector of intercepts 
        sigma: N x N covariance matrix  
    """
    X = values(frets)
    Y = values(arets)

    # Check if the length of the timeseries exceeds the lookback period
    if size(X,1) > lookback
        X = X[end-lookback+1:end,:]
        Y = Y[end-lookback+1:end,:]
    end

    # Use linear regression for parameter estimation
    mu, sigma = linreg(X, Y)

    return mu, sigma
end

#-------------------------------------------------------------------------------
# Function fmodel_rs:
#-------------------------------------------------------------------------------
function fmodel_rs(frets::TimeArray, 
    arets::TimeArray, 
    lookback::Int64,
    rsmodel::HMM)
    """
    Construct a factor model for each state in the regime-switching model (rsmodel)
        1. For each state s, perform a linear regression and estimate mu and sigma
        2. Combine the state-specific mu and sigma into a probability-weighted mu 
        and sigma using the transition probabilities conditional on the estimated 
        current state

    Inputs
        frets: T x M Timeseries of feature returns for M features and T observations.
        arets: T x N Timeseries of asset returns for N assets and T observations
        lookback: Number of observations to use to conduct the regression
        rsmodel: Object containing a T x S timeseries of smoothed probabilities for S 
            states and T observations, as well as the transition probabilities 
            conditional on the current estimated state

    Outputs
        mu: N x 1 vector of intercepts 
        sigma: N x N covariance matrix  
    """
    n_assets = size(arets, 2)
    n_states = rsmodel.n_states

    mu = [zeros(n_assets) for i=1:n_states]
    sigma = [Symmetric(zeros(n_assets, n_assets)) for i=1:n_states]

    # Use linear regression for parameter estimation per state
    states = values(rsmodel.gamma[timestamp(frets)])
    for s = 1:n_states
        X = values(frets)[states[:,s] .> 0.5, :]
        Y = values(arets)[states[:,s] .> 0.5, :]

        # Check if the length of the timeseries exceeds the lookback period
        if size(X,1) > lookback
            X = X[end-lookback+1:end,:]
            Y = Y[end-lookback+1:end,:]
        end

        mu[s], sigma[s] = linreg(X, Y)
    end

    # Calculate the probability-weighted asset covariance matrix and mean vector
    prob = rsmodel.prob
    sigma = Symmetric(sum(
                    (prob[s] .* sigma[s]) 
                    + (prob[s] .* mu[s] * mu[s]') 
                    - (prob[s] .* sum(prob[t] .* mu[s] * mu[t]' for t=1:n_states )) 
                    for s=1:n_states))
    mu = sum(prob[s] .* mu[s] for s=1:n_states)

    return mu, sigma
end

#-------------------------------------------------------------------------------
# Function linreg:
#-------------------------------------------------------------------------------
function linreg(X::Array{Float64}, Y::Matrix{Float64})
    """
    Regress matrix of targets Y against feature array X
        Note: Features are centered (de-meaned) by default

    Inputs
        Y: T x N matrix with T observations and M targets 
        X: T x M matrix with T observations and N features

    Outputs
        mu: N x 1 vector of intercepts 
        sigma: N x N covariance matrix 
    """
    n_obs, n_features = size(X)

    # Center (de-mean) the factors
    X = X .- mean(X, dims=1)
    X = [ones(n_obs, 1) X]

    # Compute the coefficients of an OLS regression
    beta = (X' * X) \ X' * Y

    # Calculate the OLS residuals
    epsilon = Y .- (X * beta)

    # Vector of intercepts (equal to the means if using centered factors)
    mu = beta[1, :]

    # Matrix of factor loadings
    V = beta[2:(n_features+1), :]

    # Factor covariance matrix
    F = cov(X[:, 2:end])

    # Diagonal matrix of residual variance
    ssq = vec(sum(epsilon .^ 2, dims=1) / (n_obs - n_features - 1))
    D = Diagonal(ssq)

    # Asset covariance matrix
    sigma = (V' * F * V) + D
    sigma = Symmetric(posdef(sigma))

    return mu, sigma
end

#-------------------------------------------------------------------------------
# Function posdef
#-------------------------------------------------------------------------------
function posdef(A::Matrix{Float64})
    """
    Find the nearest positive definite matrix to the input matrix A

    Inputs.
        A: Square matrix 
    
    Outputs.
        B: Nearest positive semidefinite matrix to A 
    """
    B = copy(A)
    counter = 0
    while minimum(real(eigvals(B))) < 0
        counter += 1
        U, S, V = svd(B)
        H = V * Diagonal(S) * V'
        B = (B + H) / 2
        B = (B + B') / 2

        if counter > 500
            println("posdef ran out of iterations to approximate the nearest 
                PSD matrix")
            break
        end
    end

    return B

end

################################################################################
# End