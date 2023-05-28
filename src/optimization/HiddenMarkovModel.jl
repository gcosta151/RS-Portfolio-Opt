# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
# 
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using TimeSeries, Statistics, Distributions, LinearAlgebra

#-------------------------------------------------------------------------------
# struct HMMparams
#-------------------------------------------------------------------------------
mutable struct HMMparams
    """
    HMMparams structure to store parameters pertaining to the HMM model
    
    Fields.
        features: List of features to be used to fit the HMM
        n_states: Number of states to which to fit the HMM
    """
    features::Vector{Symbol}
    n_states::Int64

    function HMMparams(features::Vector{Symbol}, n_states::Int64)
        return new(features, n_states)
    end
end

#-------------------------------------------------------------------------------
# struct HMM
#-------------------------------------------------------------------------------
struct HMM
    """
    Hidden Markov model (HMM) structure and constructor 
        Fit a HMM to a timeseries of feature returns by applying the Baum-Welch 
        algorithm

    Inputs.
        frets: T x M Timeseries of feature returns for M features and T observations.
            Note: The features used to fit the HMM need not be the same features 
            used to estimate the asset means and covariance matrix
        n_states: Number of states to which to fit the HMM

    Output. HMM struct with the following fields
        n_states: Number of states in the HMM. 
        gamma: T x S Timeseries of smoothed probabilities for S states with T 
            observations. The smoothed probabilities indicate the probability of
            being in state s for s = 1, ..., S
        A: S x S transition probability matrix 
        state: The esitmated state at the end of the current time period
        prob: The transition probabilities of the current state
    """
    n_states::Int
    gamma::TimeArray{Float64, 2, Date, Matrix{Float64}}
    A::Matrix{Float64}
    state::Int
    prob::Vector{Float64}

    function HMM(frets::TimeArray, n_states::Int=2)
        gamma, state, prob, A = baum_welch(frets, n_states)
        return new(n_states, gamma, A, state, prob)
    end
end

#-------------------------------------------------------------------------------
# Function baum_welch
#-------------------------------------------------------------------------------
function baum_welch(frets::TimeArray, n_states::Int64, num_iters::Int64=500)
    """
    Apply the Baum-Welch expectation-maximization algorithm to fit a HMM to the 
    given timeseries of factor returns

    Inputs. 
        frets: T x M Timeseries of feature returns for M features and T 
            observations
        n_states: Number of states to which to fit the HMM
        num_iters: Maximum number of iterations to be performed. The algorithm 
            will stop early if it converges

    Outputs.
        gamma: T x S Timeseries of smoothed probabilities for S states with T 
            observations. The smoothed probabilities indicate the probability of
            being in state s for s = 1, ..., S. 
        A: S x S transition probability matrix 
        state: The esitmated state at the end of the current time period
        prob: The transition probabilities of the current state
    """
    tstamp = timestamp(frets)
    frets = values(frets)

    # Initialize random parameters for the HMM
    A = rand(n_states, n_states) .+ 0.1
    A ./= sum(A, dims=2)
    pi = rand(n_states) .+ 0.1
    pi ./= sum(pi)

    # Initialize the mean and variance
    mu = [vec(mean(frets; dims=1)) ./ (0.5 + rand()) for i=1:n_states]
    sigma = [Symmetric(cov(frets) ./ (0.5 + rand())) for i=1:n_states]

    local gamma, A_prev, pi_prev, mu_prev, sigma_prev
    
    # Run Baum-Welch (EM) algorithm
    for i = 1:num_iters
        # E-Step       
        gamma, xi = e_step(frets, pi, A, mu, sigma)
        
        # M-Step
        pi, A, mu, sigma = m_step(frets, gamma, xi)
        
        # Check for convergence
        if i > 3
            if norm(A - A_prev) < 2e-6 && 
                norm(pi - pi_prev) < 2e-6 && 
                norm(mu - mu_prev) < 2e-6 && 
                norm(sigma - sigma_prev) < 2e-6
                break
            elseif i == num_iters
                println("HMM did not converge, consider increasing the default 
                    number of iterations")
            end
        end
        
        # Save previous parameters for convergence check
        A_prev = copy(A)
        pi_prev = copy(pi)
        mu_prev = copy(mu)
        sigma_prev = copy(sigma)
    end

    # Sort such that the first state has the lowest volatility
    idx = sortperm([tr(sigma[i]) for i=1:n_states], rev=false)
    gamma = gamma[:, idx]
    A = A[idx, :]

    # Estimated current regime and its transition probabilities
    state = findmax(gamma[end,:])[2]
    prob = vec(A[state,:])
    
    gamma = TimeArray(tstamp, gamma, ["State " * string(i) for i=1:n_states])

    return gamma, state, prob, A
end

#-------------------------------------------------------------------------------
# Function e_step
#-------------------------------------------------------------------------------
function e_step(frets::Array{Float64}, 
    pi::Vector{Float64}, 
    A::Matrix{Float64}, 
    mu::Vector{Vector{Float64}}, 
    sigma::Vector{Symmetric{Float64, Matrix{Float64}}})
    """
    Expectation step (e-step) in the Baum-Welch algorithm
        The e-step computes the expected values of the hidden state parameters.
        It estiamtes the forward (fwd) and backward (bwd) probabilities. These 
        probabilities provide estimates of the likelihood of being in a 
        particular hidden state at each time step given the observed data.

    Inputs.
        frets: T x M timeseries of feature returns for M features and T 
            observations
        pi: S x 1 vector of probabilities defining the initial state distribution
        mu: S x 1 vector where each element is a M x 1 vector of expected feature 
            returns
        sigma: S x 1 vector where each element is a M x M feature covariance 
            matrix

    Outputs.
        gamma: T x S Timeseries of smoothed (posterior) probabilities for S 
            states and T observations. The smoothed probabilities indicate the 
            probability of being in state s for s = 1, ..., S
        xi: M x M x T-1 multidimensional array of joint probabilities
            at each point in time
    """    
    n_states = size(pi,1)
    n_obs = size(frets, 1)

    fwd = zeros(n_obs, n_states)
    bwd = zeros(n_obs, n_states)
    gamma = zeros(n_obs, n_states)
    xi = zeros(n_states, n_states, n_obs-1)
    density = zeros(n_obs, n_states)
    
    # Estimate the pdf of observed data given the current estimates of mu and sigma
    for i = 1:n_states
        density[:,i] = pdf(MvNormal(mu[i], sigma[i]), frets')
    end

    # Forward pass
    fwd[1,:] = pi .* density[1,:]
    fwd[1,:] ./= sum(fwd[1,:])
    for t = 2:n_obs
        fwd[t,:] = density[t,:] .* (A' * fwd[t-1,:])
        fwd[t,:] ./= sum(fwd[t,:])
    end
    
    # Backward pass
    bwd[n_obs,:] .= density[n_obs,:]
    bwd[n_obs,:] ./= sum(bwd[n_obs,:])
    for t = n_obs-1:-1:1
        bwd[t,:] = density[t+1,:] .* (A * bwd[t+1,:])
        bwd[t,:] ./= sum(bwd[t,:])
    end

    # Compute gamma and xi
    for t = 1:n_obs
        gamma[t,:] = fwd[t,:] .* bwd[t,:]
        gamma[t,:] ./= sum(gamma[t,:])
        if t < n_obs
            xi[:,:,t] = A .* (fwd[t,:] * (bwd[t+1,:] .* density[t+1,:])')
            xi[:,:,t] ./= sum(xi[:,:,t])
        end
    end

    return gamma, xi
end

#-------------------------------------------------------------------------------
# Function m_step
#-------------------------------------------------------------------------------
function m_step(frets::Array{Float64}, 
    gamma::Matrix{Float64}, 
    xi::Array{Float64, 3})
    """
    Maximization step (m-step) in the Baum-Welch algorithm
        The m-step updates the model parameters based on the estimated expectations
        arising from the e-step. The m-step aims to maximize the likelihood of 
        the observed data. 
    
    Inputs.
        frets: T x M timeseries of feature returns for M features and T 
            observations
        gamma: T x S Timeseries of smoothed (posterior) probabilities for S states 
            and T observations. The smoothed probabilities indicate the probability 
            of being in state s for s = 1, ..., S
        xi: M x M x T-1 multidimensional array of joint probabilities
            at each point in time

    Outputs.
        pi: S x 1 vector of probabilities defining the initial state distribution
        A: S x S transition probability matrix 
        mu: S x 1 vector where each element is a M x 1 vector of expected feature 
            returns
        sigma: S x 1 vector where each element is a M x M feature covariance matrix
    """
    n_states = size(gamma,2)
    n_features = size(frets,2)

    A = zeros(n_states, n_states)
    mu = [zeros(n_features) for i=1:n_states]
    sigma = [Symmetric(zeros(n_features, n_features)) for i=1:n_states]

    # Update initial state distribution
    pi = gamma[1,:]

    for i = 1:n_states
        # Update transition matrix
        A[i,:] = sum(xi, dims=3)[i,:,1]
        A[i,:] ./= sum(A[i,:])

        # Update mean and covariance matrix for each state
        mu[i] = vec(sum(gamma[:,i] .* frets, dims=1) ./ sum(gamma[:,i]))
        diff = frets .- mu[i]'
        sigma[i] = Symmetric((diff .* gamma[:,i])' * diff ./ sum(gamma[:,i]))
    end

    return pi, A, mu, sigma
end

################################################################################
# end