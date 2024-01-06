# Regime-switching applications for portfolio optimization
#
# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
#
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using Distributions, TimeSeries, Statistics, LinearAlgebra, Plots
plotlyjs() # Use the PlotlyJS backend

include("dataload/DataLoad.jl")
import .DataLoad as dl

include("optimization/PortfolioOptimization.jl")
import .PortfolioOptimization as po

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
"""
Select the parameters for the portfolio optimization backtest
- daterange: Backtest start and end dates
- calibration: HMM calibration window (in years)
- lookback: lookback window used to calibrate the factor models (in months)
- rebalfreq: rebalance fequency (in months)
"""
daterange::Vector{Date} = [Date(2000,1,1), Date(2021,1,1)];
calibration::Year = Dates.Year(30);
lookback::Int64 = 60;
rebalfreq::Int64 = 6;

"""
Load historical data from Kenneth French's website (Fama-French data)
- Loads the three Fama-French factors
- Loads one of the following datasets as the asset data:
    - 10_Industry_Portfolios
    - 30_Industry_Portfolios
    - 49_Industry_Portfolios
Frequency can be specified as: daily, weekly or monthly
"""
data = dl.Data("30_Industry_Portfolios", daterange, calibration, "weekly");

"""
Assign the parameters to be used in the regime-switching models 
    features: List of symbols indicating the features used to fit the hidden 
        Markov model to estimate market states 
    n_states: Number of states to which to fit the HMM
"""
features::Vector{Symbol} = [Symbol("Mkt-RF")];
n_states::Int64 = 2;
rsparams = po.HMMparams(features, n_states);

#-------------------------------------------------------------------------------
# Conduct portfolio backtests
#-------------------------------------------------------------------------------
"""
models: List of optimization models to compare. Choose between
- mvo: mean-variance portfolio with a target return
- minvar: minimum variance portfolio
- rp: risk parity portfolio
- rsmvo: regime-switching mean-variance portfolio
- rsminvar: regime-switching minimum variance portfolio
- rsrp: regime-switching risk parity portfolio

constraints: List of constraints to be used in the optimization models. 
    Note: Currently, only the 'longonly' constraint is available. Removing this
    constraint will allow for short selling

portfolos: The Portfolio constructor returns a list of Portfolio-type objects. 
    Each object corresponds to the backtest of a specific model
"""
models = po.OptModel[];
constraints::Vector{Symbol} = [:longonly];
for m in [:minvar, :rsminvar, :mvo, :rsmvo, :rp, :rsrp]
    push!(models, po.OptModel(m, constraints))
end

portfolios = po.Portfolio(data, 
                            daterange, 
                            models, 
                            rsparams,
                            lookback, 
                            rebalfreq);

# Organize the output data for plotting
wealth = hcat([portfolios[i].wealth for i in 1:length(models)]...);
sharpe = hcat([portfolios[i].stats.roll_sharpe for i in 1:length(models)]...);

# Plot the wealth evolution and rolling Sharpe ratios of all portfolios
plot(wealth, 
        title="Wealth evolution", 
        linewidth=1.5, 
        xrotation = 45, 
        ylabel="Portfolio wealth")
plot(sharpe, 
        title="Rolling Sharpe ratio", 
        linewidth=1.5, 
        xrotation = 45, 
        ylabel="Portfolio Sharpe ratio")

################################################################################
# End
