module AthleteModel

using LinearAlgebra
using DifferentialEquations
using Statistics

export ModelParams, AthleteProfile, Nominal, SAE, SAS, SAR,
       RegimeSpec, regime_early_HIIT, regime_evening_intensity, regime_split_day,
       regime_alternating_hard_easy, regime_midday, regime_taper, regime_sleep_extension,
       regime_polarized,
       simulate!, Readout, readiness_series, summarize_metrics
       
export equilibrium_summary, jacobian_summary,
       weekly_averages, poincare_map, poincare_fixed_point, floquet_multipliers,
       steady_weekly_summary

include("Parameters.jl")
include("Inputs.jl")
include("ODE.jl")
include("Readiness.jl")
include("Simulation.jl")
println("AthleteModel module loaded.")

end # module
