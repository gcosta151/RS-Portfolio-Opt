# Portfolio Optimization under a Markov Regime-Switching Framework #
 This repository contains the source code to conduct numerical experiments similar to those presented in the following two papers:
 - [Costa, G. and Kwon, R. H. (2019). Risk parity portfolio optimization under a Markov regime-switching framework. Quantitative Finance, 19(3), 453-47](https://www.tandfonline.com/doi/abs/10.1080/14697688.2018.1486036?journalCode=rquf20)
    - [Link to PDF](https://www.researchgate.net/profile/Giorgio-Costa-2/publication/326756996_Risk_parity_portfolio_optimization_under_a_Markov_regime-switching_framework/links/5e0992d74585159aa4a47d19/Risk-parity-portfolio-optimization-under-a-Markov-regime-switching-framework.pdf)
 - [Costa, G. and Kwon, R. H. (2020). A regime-switching factor model for mean–variance optimization. Journal of Risk, 22(4), 31-59](https://www.risk.net/journal-of-operational-risk/7535001/a-regime-switching-factor-model-for-mean-variance-optimization)
    - [Link to PDF](https://www.researchgate.net/profile/Giorgio-Costa-2/publication/341752309_A_Regime-Switching_Factor_Model_for_Mean-Variance_Optimization/links/61ddd756323a2268f9997b5f/A-Regime-Switching-Factor-Model-for-Mean-Variance-Optimization.pdf)
 
The work in these two papers presents a Markov regime-switching factor model to describe the cyclical nature of asset returns in modern financial markets. Maintaining a factor model structure allows us to easily derive the first two moments of the asset return distribution: the expected returns and covariance matrix. By design, these two parameters are calibrated under the assumption of having distinct market regimes. In turn, these regime-dependent parameters serve as the inputs during portfolio optimization, thereby constructing portfolios adapted to the current market environment. The proposed framework leads to a computationally tractable portfolio optimization problem, meaning we can construct large, realistic portfolios. 

## Dependencies ##
- Julia v1.x
- JuMP.jl v1.x
- Ipopt.jl v1.x
- TimeSeries.jl v0.23

## Usage ##
This repository contains all the files used to necessary to run the numerical experiments of the regime-switching portfolio optimization framework. To run the experiments. please refer to the main.jl file. Anyone wishing to make any changes to the models do so by tinkering with a copy of the code base. The code base is made up of the following files:
- dataload/DataLoad.jl: Module to download data from [Kenneth French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Use this module to download returns of the Fama-French three-factor model, as well as the Industry Portfolios to serve as the historical asset returns. 
- optimization/PortfolioOptimization.jl: Module to construct optimal nominal and regime-switching portfolios. Six portfolio optimization models are currently available for use:
 - mvo: Nominal mean-variance optimization
 - rsmvo: Regime-switching mean variance optimization
 - minvar: Nominal minimum variance optimization
 - rsminvar: Regime-switching minimum variance optimization
 - rp: Risk parity portfolio optimization
 - rsrp: Regime-switching risk parity portfolio optimization.
- optimization/Optimization.jl: Supporting script called by the PortfolioOptimization.jl module. This script contains the JuMP-based optimization models.
- optimization/HiddenMarkovModel.jl: Supporting script called by the PortfolioOptimization.jl module. This script contains an implementation of the Baum-Welch algorithm to fit a hidden Markov model to the factor returns. 
- optimization/FactorModels.jl: Supporting script called by the PortfolioOptimization.jl module. This script contains an implementation of linear regression under a single regime, as well as under the assumption of multiple regimes. 

# Licensing
Unless otherwise stated, the source code is copyright of Giorgio Costa and licensed under the Apache 2.0 License.


 
