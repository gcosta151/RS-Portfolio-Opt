# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
# 
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using Statistics, LinearAlgebra, JuMP, Ipopt

#-------------------------------------------------------------------------------
# struct OptModel
#-------------------------------------------------------------------------------
mutable struct OptModel
    """
    HMMparams structure to store parameters pertaining to the HMM model
    
    Fields.
        model: Symbol indicating the selected portfolio optimization model
        constraints: Set of constraints to be used by the optimization model
        solver: Solver used in optimization. Default solver is Ipopt
    """
    model::Symbol
    constraints::Vector{Symbol}
    solver::DataType

    function OptModel(model::Symbol, constraints::Vector{Symbol})
        return new(model, constraints, Ipopt.Optimizer)
    end
end

#-------------------------------------------------------------------------------
# Function mvo_core
#-------------------------------------------------------------------------------
function minvar_core(sigma::Symmetric{Float64, Matrix{Float64}}, 
    optparams::OptModel)
    """
    Minimum variance optimization model 

    minimize    w' Q w 
    s.t.        sum(w) = 1
                w >= 0 (if long-only constraint selected)

    Note: The target return constraint assumes the target is 5% above the 
        average of the asset mean returns. 

    Inputs. 
        sigma: n x n asset covariance matrix
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 

    Output.
        w: Optimal portfolio weights
    """
    n = size(sigma,1)
  
    # Optimization model
    m = Model(
        optimizer_with_attributes(optparams.solver, "tol"=>1e-14, "print_level"=>0)
        )
  
    # Portfolio weight (decision variable)
    if :longonly in optparams.constraints
        @variable(m, w[1:n] >= 0)
    else
        @variable(m, w[1:n])
    end
  
    # Budget constraint
    @constraint(m, sum(w) == 1)
  
    # Objective: minimize portfolio variance
    @objective(m, Min, w' * sigma * w)
    optimize!(m)
  
    return value.(w)  
end


#-------------------------------------------------------------------------------
# Function mvo_core
#-------------------------------------------------------------------------------
function mvo_core(mu::Vector{Float64}, 
    sigma::Symmetric{Float64, Matrix{Float64}}, 
    optparams::OptModel)
    """
    Mean-variance optimization model 

    minimize    w' Q w 
    s.t.        mu' w >= 1.05 * avg(mu)
                sum(w) = 1
                w >= 0 (if long-only constraint selected)

    Note: The target return constraint assumes the target is 5% above the 
        average of the asset mean returns. 

    Inputs. 
        mu: n x 1 vector of asset means 
        sigma: n x n asset covariance matrix
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 

    Output.
        w: Optimal portfolio weights
    """
    n = size(mu,1)

    # Set the target return
    target = 1.05 * mean(mu)
  
    # Optimization model
    m = Model(
        optimizer_with_attributes(optparams.solver, "tol"=>1e-14, "print_level"=>0)
        )
  
    # Portfolio weight (decision variable)
    if :longonly in optparams.constraints
        @variable(m, w[1:n] >= 0)
    else
        @variable(m, w[1:n])
    end
  
    # Budget and target return constraints
    @constraint(m, sum(w) == 1)
    @constraint(m, mu' * w >= target)
  
    # Objective: minimize portfolio variance
    @objective(m, Min, w' * sigma * w)
    optimize!(m)
  
    return value.(w)  
end

#-------------------------------------------------------------------------------
# Function rp_core
#-------------------------------------------------------------------------------
function rp_core(sigma::Symmetric{Float64, Matrix{Float64}}, 
    optparams::OptModel)
    """
    Risk parity portfolio optimization model 

    minimize    w' Q w - z
    s.t.        z <= sum(log(w[i]) for i=1:n)
                w >= 0

    Note: The risk parity model assumes that positions can only be held long. 
        This assumption allows us to model risk parity as a convex problem.  

    Inputs. 
        sigma: n x n asset covariance matrix
        optparams: OptModel struct specifying the set of constraints and solver to 
            use in the optimization models 

    Output.
        w: Optimal portfolio weights
    """
    n = size(sigma,1)

    # Optimization model
    m = Model(
        optimizer_with_attributes(optparams.solver, "tol"=>1e-14, "print_level"=>0)
        )
  
    # w: Portfolio weight (decision variable), z: auxiliary variable
    @variable(m, w[1:n] >= 0)
    @variable(m, z)
  
    # Auxiliary log-barrier non-linear constraint 
    @NLconstraint(m, z <= sum( log(w[i]) for i = 1:n ))
  
    # Objective: minimize portfolio variance
    @objective(m, Min, w' * sigma * w - z)
    optimize!(m)
  
    # Normalize weights
    w = value.(w) ./ sum(value.(w))

    return w
end

################################################################################
# End