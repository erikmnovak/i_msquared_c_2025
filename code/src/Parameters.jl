# Deterministic parameter sets (Nominal + 3 synthetic athletes)
# All units are as in the paper: time → days; B kernel uses hours but is converted to days internally.

#= ---------------------------------------------------------------------------
ModelParams: container of all parameters used in the six ODEs and helpers.
This is intentionally explicit (no Dicts) for readability and performance.
Fields mirror Section 6 of the paper (Nominal values).
--------------------------------------------------------------------------- =#

Base.@kwdef mutable struct ModelParams
    # ---- Top row: capacities A, N
    kA::Float64 = 0.035
    αE::Float64 = 0.60
    αH::Float64 = 0.40
    KA::Float64 = 1.0
    τA::Float64 = 50.0
    θAI::Float64 = 0.030
    cS::Float64 = 0.50
    cC::Float64 = 0.40

    kN::Float64 = 0.040
    αS::Float64 = 0.70
    αHN::Float64 = 0.30
    KN::Float64 = 1.0
    τN::Float64 = 35.0
    θNI::Float64 = 0.030
    μE::Float64 = 0.30

    # ---- Middle row: fatigue
    γE::Float64 = 0.55
    γH::Float64 = 0.90
    γS::Float64 = 0.45
    τfa::Float64 = 1.0
    ρa::Float64 = 0.50

    ε::Float64 = 0.18
    ξc::Float64 = 0.20
    τfc::Float64 = 10.0
    ρc::Float64 = 0.40

    # ---- Bottom row: sleep S and damage I
    λw::Float64 = 0.30
    λT::Float64 = 0.20
    ξs::Float64 = 0.15
    τS::Float64 = 6.0
    μs::Float64 = 0.30

    ψ0::Float64 = 0.40
    κE::Float64 = 0.30
    κH::Float64 = 0.35
    κS::Float64 = 0.35
    χa::Float64 = 0.25
    χc::Float64 = 0.25
    τI::Float64 = 14.0
    ψs::Float64 = 0.25
    ψn::Float64 = 0.20

    # ---- Sleep efficiency and bedtime kernel
    q0::Float64 = 0.88
    η::Float64  = 1.00
    βE::Float64 = 0.30
    βH::Float64 = 0.40
    βS::Float64 = 0.30
    σb::Float64 = 1.5    # hours (decay)
    hb::Float64 = 6.0    # hours (horizon)

    # ---- Readiness weights
    w_end::Float64 = 0.50
    w_str::Float64 = 0.50
    λa::Float64 = 0.30
    λc::Float64 = 0.50
    λs::Float64 = 0.40
    λi::Float64 = 0.60

    # ---- Initial conditions
    A0::Float64 = 0.65
    N0::Float64 = 0.65
    Fa0::Float64 = 0.20
    Fc0::Float64 = 0.20
    S0::Float64  = 0.20
    I0::Float64  = 0.15
end

# Pretty name for a parameterized athlete
struct AthleteProfile
    name::String
    p::ModelParams
end

# ---- Nominal (all‑round) athlete
Nominal() = AthleteProfile("Nominal", ModelParams())

# ---- Synthetic athletes (deterministic overrides relative to Nominal)

function SAE()
    p = ModelParams()
    p.kA  = 0.045
    p.αE  = 0.70
    p.αH  = 0.30
    p.kN  = 0.035
    p.τfa = 0.9
    p.ρa  = 0.60
    p.τfc = 9.0
    p.ρc  = 0.50
    p.μE  = 0.18
    p.η   = 0.70
    p.w_end = 0.65
    p.w_str = 0.35
    return AthleteProfile("SA‑E (Aerobic responder)", p)
end

function SAS()
    p = ModelParams()
    p.kN   = 0.055
    p.αS   = 0.80
    p.αHN  = 0.20
    p.kA   = 0.030
    p.μE   = 0.55
    p.γS   = 0.60
    p.κS   = 0.45
    p.ψ0   = 0.45
    p.χa   = 0.30
    p.χc   = 0.30
    p.w_end = 0.35
    p.w_str = 0.65
    return AthleteProfile("SA‑S (Neuromuscular responder)", p)
end

function SAR()
    p = ModelParams()
    p.cS   = 0.80
    p.cC   = 0.60     
    p.τfa  = 1.3
    p.ρa   = 0.35
    p.τfc  = 14.0
    p.ρc   = 0.30
    p.τS   = 8.0
    p.μs   = 0.25
    p.ξs   = 0.25
    p.η    = 1.50
    p.q0   = 0.82
    p.ψs   = 0.20
    p.ψn   = 0.15
    p.λs   = 0.60
    p.λi   = 0.70
    p.S0   = 0.30
    return AthleteProfile("SA‑R (Recovery‑limited)", p)
end

# Export the factories from the module:
export Nominal, SAE, SAS, SAR
