# DataLoad module
#
# Prepared by:    Giorgio Costa
# Last revision:  27-May-2023
#
################################################################################
module DataLoad

export Data

#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
using TimeSeries, Dates, CSV, DataFrames, HTTP, ZipFile

#-------------------------------------------------------------------------------
# struct Data
#-------------------------------------------------------------------------------
struct Data
    """
    Data structure and constructor
        Specify the datasets to be downloaded from Kenneth French's website and 
        subset the desired date range. The Fama-French three factors are loaded
        as the features by default. 
        
    Inputs. 
        dataset: String specifying the asset dataset to be used
        daterange: 2 x 1 vector of dates specifying the start and end date of the
            historical backtest 
        calibration: Number of years by which to extend the start of the
            historical dataset. The additional time period is used for model 
            training and calibration
        freq: Frequency of the observations in the time series. Can be set to 
            "monthly", "weekly" or "daily"

    Outputs. Data struct with the following fields
        arets: T x N timeseries of asset returns for N assets and T observations
        prices: T x N timeseries of asset prices for N assets and T observations
        frets: T x M timeseries of feature returns for M features and T observations
        rf: T x 1 timeseries of the risk-free rate for T observations
        dataset: String specifying the asset dataset used 
        freq: Frequency of the observations in the time series. Can be set to 
            "monthly", "weekly" or "daily"
    """
    arets::TimeArray{Float64}
    prices::TimeArray{Float64}
    frets::TimeArray{Float64}
    rf::TimeArray{Float64}
    dataset::String
    freq::String

    function Data(dataset::String, 
        daterange::Vector{Date}, 
        calibration::Year=Dates.Year(0), 
        freq::String="monthly")

        sdate = daterange[1]
        edate = daterange[2]
        arets, prices, frets, rf = famafrenchdata(dataset, 
                                                    daterange,
                                                    calibration,
                                                    freq)
        sdate = sdate - calibration

        return new(arets[sdate:edate], 
                    prices[sdate:edate], 
                    frets[sdate:edate], 
                    rf[sdate:edate], 
                    dataset, 
                    freq)
    end
end

#-------------------------------------------------------------------------------
# Function famafrenchdata
#-------------------------------------------------------------------------------
function famafrenchdata(dataset::String, 
    daterange::Vector{Date}, 
    calibration::Year,
    freq::String)
    """
    This function loads Fama-French data from Kenneth French's data library:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
        - The Fama-French three factor model is loaded by default as the feature 
            returns.
        - The Industry Portfolios are used as a proxy for asset data, with each 
            Industry akin to a representative ETF of such industry. 

    Inputs. 
        dataset: String specifying the asset dataset to be used
        daterange: 2 x 1 vector of dates specifying the start and end date of the
            historical backtest 
        calibration: Number of years by which to extend the start of the
            historical dataset. The additional time period is used for model 
            training and calibration
        freq: Frequency of the observations in the time series. Can be set to 
            "monthly", "weekly" or "daily"

    Outputs.
        arets: T x N timeseries of asset returns for N assets and T observations
        prices: T x N timeseries of asset prices for N assets and T observations
        frets: T x M timeseries of feature returns for M features and T observations
        rf: T x 1 timeseries of the risk-free rate for T observations
    """
    @assert dataset in ["10_Industry_Portfolios", 
        "30_Industry_Portfolios", 
        "49_Industry_Portfolios"] "dataset must be one of the following: 
        10_Industry_Portfolios, 30_Industry_Portfolios, 49_Industry_Portfolios"

    @assert freq in ["monthly", 
        "weekly", 
        "daily"] "freq must be one of the following: monthly, weekly or daily"

    if freq == "monthly"
        freq_string = "" 
    else 
        freq_string = "_" * freq
    end

    # Download factor data
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors" * freq_string * "_CSV.zip"
    response = HTTP.get(url)
    io = IOBuffer(response.body)
    z = ZipFile.Reader(io)
    ff = first(z.files)
    stringarray = readlines(ff, keep=true)
    ios = IOBuffer(string(stringarray...))
    if freq == "monthly"
        frets = CSV.read(ios, DataFrame; header=3, silencewarnings=true)
        idx_end = findfirst(ismissing, frets[:, :Column1]) - 2
        frets = frets[1:idx_end, :]
        tstamp = frets[:,1]
        tstamp = Date.(string.(tstamp), "yyyymm")
    else
        frets = CSV.read(ios, DataFrame; header=4, footerskip=1, silencewarnings=true)
        tstamp = frets[:,1]
        tstamp = Date.(string.(tstamp), "yyyymmdd")
    end
    fnames = string.(names(frets))[2:end]
    if eltype(frets[:,2]) <: Float64
        frets = Matrix( frets[:, 2:end] ) ./ 100
    else
        frets = Matrix( parse.(Float64, frets[:, 2:end]) ) ./ 100
    end
    frets = TimeArray( tstamp, frets, Symbol.(fnames) )[Date(1970,1,1):daterange[2]]
    
    # Download asset data
    if freq == "weekly"
        freq_string = "_daily"
    end
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" * dataset * freq_string * "_CSV.zip"
    response = HTTP.get(url)
    io = IOBuffer(response.body)
    z = ZipFile.Reader(io)
    ff = first(z.files)
    stringarray = readlines(ff, keep=true)
    ios = IOBuffer(string(stringarray...))
    if freq == "monthly"
        arets = CSV.read(ios, DataFrame; header=12, silencewarnings=true)
        idx_end = findfirst(ismissing, arets[:, :Column1]) - 2
        arets = arets[1:idx_end, :]
        tstamp = arets[:,1]
        tstamp = Date.(string.(tstamp), "yyyymm")
    else
        arets = CSV.read(ios, DataFrame; header=10, footerskip=1, silencewarnings=true)
        idx_end = findfirst(ismissing, arets[:, :Column1]) - 2
        arets = arets[1:idx_end, :]
        tstamp = arets[:,1]
        tstamp = Date.(string.(tstamp), "yyyymmdd")
    end
    anames = string.(names(arets))[2:end]
    if eltype(arets[:,2]) <: Float64
        arets = Matrix( arets[:, 2:end] ) ./ 100
    else
        arets = Matrix( parse.(Float64, arets[:, 2:end]) ) ./ 100
    end
    arets = TimeArray( tstamp, arets, Symbol.(anames) )[Date(1969,11,1):daterange[2]]
    
    # Calculate asset prices
    prices = cumprod(values(arets) .+ 1, dims=1) ./ (values(arets)[1,:] .+ 1)'
    prices = TimeArray( timestamp(arets), prices, Symbol.(anames) )
    if freq == "weekly"
        prices = collapse(prices, week, last) 
        arets = percentchange(prices)
    end
    prices = prices[daterange[1]-calibration:daterange[2]]

    # Calculate asset excess returns, risk-free rate and factors
    rf = frets[:RF][daterange[1]-calibration:daterange[2]]
    arets = arets[daterange[1]-calibration:daterange[2]]
    arets = TimeSeries.rename(arets .- frets[:RF], Symbol.(anames))
    frets = frets[Symbol.(fnames[1:end-1])][daterange[1]-calibration:daterange[2]]
    
    return arets, prices, frets, rf
end

################################################################################
# Module end
end
