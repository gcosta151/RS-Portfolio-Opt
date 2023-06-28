# Optimization module
#
# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
#
################################################################################
module PortfolioOptimization

export Portfolio, Stats, HMM, HMMparams, OptModel

#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using TimeSeries, Statistics, Distributions, LinearAlgebra
import ..DataLoad as dl

include("HiddenMarkovModel.jl")
include("FactorModels.jl")
include("Optimization.jl")

#-------------------------------------------------------------------------------
# struct Stats
#-------------------------------------------------------------------------------
struct Stats
    """
    Stats structure and constructor 
        Calculate the summary statistics of a portfolio backtest

    Inputs.
        wealth: Timeseries with the portfolio wealth evolution during the 
            backtest
        lookback: Number of observations used to compute the rolling Sharpe ratio
        freq: Frequency of observations

    Output. Stats struc with the following fields
        mu: Average portfolio return (annualized)
        vol: Average portfolio volatility (annualized)
        sharpe: Average portfolio Sharpe ratio (annualized)
        roll_sharpe: Rolling Sharpe ratio based on the lookback window 
            (annualized)
    
    Note: The Sharpe ratio here is simplified as the ratio of mu / vol and it
        does not adjust for the risk-free rate. 
    """
    mu::Float64
    vol::Float64
    sharpe::Float64
    roll_sharpe::TimeArray{Float64, 1, Date, Vector{Float64}}

    function Stats(wealth::TimeArray{Float64, 1, Date, Vector{Float64}}, 
                    lookback::Int64, 
                    freq::String)

        if freq == "daily"
            freq = 252
        elseif freq == "weekly"
            freq = 52
        elseif freq == "monthly"
            freq = 12
        end

        rets = percentchange(wealth)
        mu = (prod(values(rets) .+ 1) ^ (1/length(rets))) ^ freq - 1
        vol = std(values(rets)) * sqrt(freq)
        sharpe = mu / vol

        roll_sharpe = moving(x -> (
            (prod(values(x) .+ 1) ^ (1/length(x))) ^ freq - 1) / std(x), 
            rets, lookback)

        new(mu, vol, sharpe, roll_sharpe)
    end
end

#-------------------------------------------------------------------------------
# struct Portfolio
#-------------------------------------------------------------------------------
struct Portfolio
    """
    Portfolio structure and constructor 
        Conduct a historical backtest using the data provided for the specified 
        models

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        daterange: 2-element vector containing the start and end date of the 
            backtest 
        models: Vector of symbols that specify the models to be tested
        constraints: Set of constraints to be used by the optimization models
        flist_rs: List of factors to be used to fit the HMM
        n_states: Number of states to which to fit the HMM
        lookback: Number of months to use in the regression models for parameter 
            estimation
        rebalfreq: Number of months between portfolio rebalancing periods

    Outputs. List of portfolio structs. Each portfolio has the following fields
        optparams: OptModel struct specifying the optimization model used and 
            the set of constraints applied, as well as the solver used.
        rsparams: HMMparams struct containing the list of features and number of 
            states used to fit the HMM
        wealth: Timeseries with the portfolio wealth evolution during the backtest
        n_assets: Number of assets N used in portfolio construction
        n_features: Number of features M used in portfolio construction
        lookback: Number of months used in the regression models for parameter 
            estimation
        rebalfreq: Number of months between portfolio rebalancing periods
        stats: Stats struct containing the portfolio summary statistics
    """
    optparams::OptModel
    rsparams::HMMparams
    wealth::TimeArray{Float64, 1, Date, Vector{Float64}}
    n_assets::Int64
    n_features::Int64
    lookback::Int64
    rebalfreq::Int64
    stats::Stats

    function Portfolio(data::dl.Data, 
        daterange::Vector{Date},
        models::Vector{OptModel}, 
        rsparams::HMMparams,
        lookback::Int64=36, 
        rebalfreq::Int64=6)

        # If necessary, convert the lookback window from months to weeks or days
        if data.freq == "monthly"
            lback = lookback
        elseif data.freq == "weekly"
            # There are approximately 4.33 weeks per month
            lback = Int64(round(lookback * 4.33))
        elseif data.freq == "daily"
            # There are approximately 21 trading days per month
            lback = lookback * 21
        end

        n_models = length(models)
        n_assets = size(data.arets,2)
        n_features = size(data.frets,2)
        wealth = backtest(data, 
                        daterange,
                        models,
                        rsparams,
                        lback,
                        rebalfreq)

        return [new(models[m], 
                    rsparams, 
                    wealth[m], 
                    n_assets, 
                    n_features,
                    lookback, 
                    rebalfreq,
                    Stats(wealth[m], lback, data.freq)) for m = 1:n_models]
    end
end

#-------------------------------------------------------------------------------
# Function backtest
#-------------------------------------------------------------------------------
function backtest(data::dl.Data, 
    daterange::Vector{Date},
    models::Vector{OptModel},
    rsparams::HMMparams, 
    lookback::Int64,
    rebalfreq::Int64)
    """
    The backtest function conducts a historical backtest of the selected portfolio
        optimization model over the desired date range with periodic rebalancing

    Inputs. 
        data: Data object containing the feature and asset historical prices and 
            returns
        daterange: 2-element vector containing the start and end date of the 
            backtest 
        models: Vector of OptModel structs that specify the models to be tested
        rsparams: HMMparams struct containing the list of features and number of 
            states used to fit the HMM
        lookback: Number of months to use in the regression models for parameter 
            estimation
        rebalfreq: Number of months between portfolio rebalancing periods

    Outputs.
        wealth: Timeseries with the portfolio wealth evolution during the backtest
    """
    daterange = daterange[1]:Month(rebalfreq):daterange[2]
    n_models = length(models)
    wealth = [[100.0] for i = 1:n_models]
    
    for (sdate, edate) in zip(daterange[1:end-1], daterange[2:end] .- Day(1))

        # Fit the HMM
        rsmodel = HMM(data.frets[Date(1,1,1):(sdate-Day(1))][rsparams.features], rsparams.n_states)

        # Portfolio optimization
        for m = 1:n_models
            if models[m].model in [:rsmvo, :rsminvar, :rsrp]
                w = eval(models[m].model)(data, (sdate-Day(1)), models[m], lookback, rsmodel)
            else
                w = eval(models[m].model)(data, (sdate-Day(1)), models[m], lookback)
            end

            cumrets = wealth[m][end] .* cumprod(
                values(data.arets[sdate:edate] .+ data.rf[sdate:edate]) .+ 1, dims=1)
                
            append!(wealth[m], sum(cumrets .* w', dims=2))
        end
        
    end

    tstamp = timestamp(data.frets)[end-length(wealth[1])+1:end]        
    wealth = [TimeArray(tstamp, wealth[m], [models[m].model]) for m = 1:n_models]

    return wealth
end

#-------------------------------------------------------------------------------
# Function mvo
#-------------------------------------------------------------------------------
function mvo(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64)
    """
    The mvo function performs mean-variance optimization by using a standard 
        factor model to estimate the asset means and covariance matrix, and then
        calling the mvo_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 
        lookback: Number of months to use in the regression models for parameter 
            estimation

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel(data.frets[Date(1,1,1):edate], 
                        data.arets[Date(1,1,1):edate], 
                        lookback)
    
    w = mvo_core(mu, sigma, optparams)
  
    return w
end

#-------------------------------------------------------------------------------
# Function rsmvo
#-------------------------------------------------------------------------------
function rsmvo(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64, 
    rsmodel::HMM)
    """
    The rsmvo function performs mean-variance optimization by using a regime-
        switching factor model to estimate the asset means and covariance matrix, 
        and then calling the mvo_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 
        lookback: Number of months to use in the regression models for parameter 
            estimation
        rsmodel: Object containing a T x S timeseries of smoothed probabilities 
            for S states and T observations, as well as the transition 
            probabilities conditional on the current estimated state

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel_rs(data.frets[Date(1,1,1):edate], 
                            data.arets[Date(1,1,1):edate],
                            lookback,
                            rsmodel)
    
    w = mvo_core(mu, sigma, optparams)
  
    return w
end

#-------------------------------------------------------------------------------
# Function rp
#-------------------------------------------------------------------------------
function rp(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64)
    """
    The rp function performs risk parity portfolio optimization by using a 
        standard factor model to estimate the asset means and covariance matrix, 
        and then calling the rp_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 
        lookback: Number of months to use in the regression models for parameter 
            estimation

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel(data.frets[Date(1,1,1):edate], 
                        data.arets[Date(1,1,1):edate], 
                        lookback)
    
    w = rp_core(sigma, optparams)

    return w
end

#-------------------------------------------------------------------------------
# Function rsrp
#-------------------------------------------------------------------------------
function rsrp(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64, 
    rsmodel::HMM)
    """
    The rsrp function performs risk parity portfolio optimization by using a 
        regime-switching factor model to estimate the asset means and covariance 
        matrix, and then calling the rp_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 
        lookback: Number of months to use in the regression models for parameter 
            estimation
        rsmodel: Object containing a T x S timeseries of smoothed probabilities 
            for S states and T observations, as well as the transition 
            probabilities conditional on the current estimated state

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel_rs(data.frets[Date(1,1,1):edate], 
                            data.arets[Date(1,1,1):edate], 
                            lookback,
                            rsmodel)
    
    w = rp_core(sigma, optparams)

    return w
end

#-------------------------------------------------------------------------------
# Function minvar
#-------------------------------------------------------------------------------
function minvar(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64)
    """
    The minvar function performs minimum variance optimization by using a standard 
        factor model to estimate the asset means and covariance matrix, and then
        calling the minvar_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 
        lookback: Number of months to use in the regression models for parameter 
            estimation

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel(data.frets[Date(1,1,1):edate], 
                        data.arets[Date(1,1,1):edate], 
                        lookback)
    
    w = minvar_core(sigma, optparams)
  
    return w
end

#-------------------------------------------------------------------------------
# Function rsminvar
#-------------------------------------------------------------------------------
function rsminvar(data::dl.Data, 
    edate::Date, 
    optparams::OptModel,
    lookback::Int64, 
    rsmodel::HMM)
    """
    The rsminvar function performs minimum variance optimization by using a regime-
        switching factor model to estimate the asset means and covariance matrix, 
        and then calling the minvar_core function for portfolio optimization

    Inputs.
        data: Data object containing the feature and asset historical prices and 
            returns
        edate: Date before the start of the current rebalancing period. Indicates
            the end date of the training period
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models  
        lookback: Number of months to use in the regression models for parameter 
            estimation
        rsmodel: Object containing a T x S timeseries of smoothed probabilities 
            for S states and T observations, as well as the transition 
            probabilities conditional on the current estimated state

    Outputs.
        w: N x 1 vector of optimal portfolio weights for N assets
    """
    mu, sigma = fmodel_rs(data.frets[Date(1,1,1):edate], 
                                data.arets[Date(1,1,1):edate], 
                                lookback,
                                rsmodel)
    
    w = minvar_core(sigma, optparams)
  
    return w
end

################################################################################
# Module end
end