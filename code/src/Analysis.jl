# src/Analysis.jl
using LinearAlgebra, ForwardDiff, NLsolve
# Reuse code from your module
# - ModelParams, AthleteProfile
# - B_of_t_factory, q_of_B_factory
# - rhs!, readiness, readiness_series, summarize_metrics
# - simulate! (we’ll also add a short-interval solver below)

# ---------- A. Weekly averages for stationary surrogate ----------
"""
    weekly_averages(ath, rs; T=7.0, dt=1/96)

Return a NamedTuple of weekly averages:
:ubarE, :ubarH, :ubarS, :sbar, :sqbar, :nbar, :xbar
where `sqbar = mean(s(t)*q(B(t)))` is the clearance-effective quantity.
"""
function weekly_averages(ath::AthleteProfile, rs; T=7.0, dt=1/96)
    p = ath.p
    B = B_of_t_factory(rs.uE, rs.uH, rs.uS, p, rs)
    q = q_of_B_factory(p)(B)
    ts = 0.0:dt:T
    ubarE = mean(rs.uE.(ts)); ubarH = mean(rs.uH.(ts)); ubarS = mean(rs.uS.(ts))
    sbar  = mean(rs.s.(ts))
    sqbar = mean(rs.s.(ts) .* q.(ts))     # what multiplies clearances at night
    nbar  = mean(rs.n.(ts)); xbar = mean(rs.x.(ts))
    return (ubarE=ubarE, ubarH=ubarH, ubarS=ubarS, sbar=sbar, sqbar=sqbar, nbar=nbar, xbar=xbar)
end

# ---------- B. Stationary surrogate RHS (constant inputs) ----------
"""
    rhs_const!(du, u, p, means)

RHS with constant inputs given by `means` from `weekly_averages`.
`means` must define: ubarE, ubarH, ubarS, sbar, sqbar, nbar, xbar.
"""
function rhs_const!(du, u, p::ModelParams, means)
    # unpack
    m = means
    # Build constant callables inline for gates that depend on inputs only
    s = m.sbar; q = m.sqbar; n = m.nbar; x = m.xbar
    uE = m.ubarE; uH = m.ubarH; uS = m.ubarS

    # states
    A,N,Fa,Fc,S,I = u

    # Gates
    Grec = 1.0 / (1.0 + p.cS*S + p.cC*Fc)
    Gint = 1.0 / (1.0 + p.μE*uE)

    # ---- Top row
    dA = p.kA*(p.αE*uE + p.αH*uH)*Grec*(1.0 - A/p.KA) - A/p.τA - p.θAI*I
    dN = p.kN*(p.αS*uS + p.αHN*uH)*Grec*Gint*(1.0 - N/p.KN) - N/p.τN - p.θNI*I

    # ---- Middle row
    dFa = (p.γE*uE + p.γH*uH + p.γS*uS) - (1.0/p.τfa + p.ρa*s*q)*Fa
    dFc = p.ε*Fa + p.ξc*x - (1.0/p.τfc + p.ρc*s*q)*Fc

    # ---- Bottom row
    dS  = p.λw*(1.0 - s) + p.λT*(p.γE*uE + p.γH*uH + p.γS*uS) + p.ξs*x - (1.0/p.τS + p.μs*s*q)*S
    load_to_I = p.ψ0*(p.κE*uE + p.κH*uH + p.κS*uS)*(1.0 + p.χa*Fa + p.χc*Fc)
    dI  = load_to_I - (1.0/p.τI + p.ψs*s*q + p.ψn*n)*I

    du[1]=dA; du[2]=dN; du[3]=dFa; du[4]=dFc; du[5]=dS; du[6]=dI
    return nothing
end

# Convenience: vector field used by ForwardDiff
rhs_const_vec(u, p, means) = (du = similar(u); rhs_const!(du, u, p, means); du)

# ---------- C. Equilibrium & Jacobian ----------
"""
    equilibrium_summary(ath, rs; guess=nothing)

Compute weekly averages, then solve f(z, means)=0 (stationary surrogate).
Returns (zstar, Pstar, J, eigvals).
"""
function equilibrium_summary(ath::AthleteProfile, rs; guess=nothing)
    p = ath.p
    means = weekly_averages(ath, rs)
    z0 = isnothing(guess) ? [p.A0,p.N0,p.Fa0,p.Fc0,p.S0,p.I0] : copy(guess)

    F(z) = rhs_const_vec(z, p, means)
    # NLsolve on a 6D root
    sol = nlsolve(z -> F(z), z0; autodiff=:forward)
    zstar = sol.zero
    # Jacobian via ForwardDiff
    J = ForwardDiff.jacobian(z -> rhs_const_vec(z, p, means), zstar)
    # Readiness
    A,N,Fa,Fc,S,I = zstar
    Pstar = readiness(A,N,Fa,Fc,S,I,p)
    return (zstar=zstar, Pstar=Pstar, J=J, eig=eigvals(J), means=means)
end

"""
    jacobian_summary(J)

Return sorted eigenvalues and derived half-life (days) from dominant real part.
"""
function jacobian_summary(J)
    λ = eigvals(J)
    λr = real.(λ)
    λmax = maximum(λr)
    halflife = (λmax < 0) ? log(2)/abs(λmax) : Inf
    return (eig=λ, λmax=λmax, halflife=halflife)
end

# ---------- D. Poincaré map & Floquet multipliers (weekly) ----------
# Short-interval solver with custom initial state and regime
function _integrate_interval(u0, ath::AthleteProfile, rs; T=7.0, saveat=1/24)
    # reuse Simulation.jl machinery
    p = ath.p
    B = B_of_t_factory(rs.uE, rs.uH, rs.uS, p, rs)
    q = q_of_B_factory(p)(B)
    inputs = (uE=rs.uE, uH=rs.uH, uS=rs.uS, s=rs.s, n=rs.n, x=rs.x, B=B, q=q)
    prob = ODEProblem((du,u,pp,t)->rhs!(du,u,p,t,inputs), u0, (0.0,T))
    sol = solve(prob, DP5(); saveat=saveat, reltol=1e-6, abstol=1e-6, maxiters=1_000_000)
    return sol
end

"""
    poincare_map(z0, ath, rs; T=7.0)

One-week map Φ_T(z0) produced by integrating the full nonstationary ODE.
"""
function poincare_map(z0, ath, rs; T=7.0)
    sol = _integrate_interval(z0, ath, rs; T=T, saveat=T)
    return Array(sol.u[end])  # state at T
end

"""
    poincare_fixed_point(ath, rs; T=7.0, warmup_weeks=6)

Find z* = Φ_T(z*) using NLsolve on G(z)=Φ_T(z)-z.
Warm up by simulating several weeks to get a good initial guess.
"""
function poincare_fixed_point(ath, rs; T=7.0, warmup_weeks=6)
    # warmup to near-cycle
    rd = simulate!(ath, rs; T_days=warmup_weeks*T, saveat=1/24)
    z0 = rd.states[:,end]   # last state as starting guess
    G(z) = poincare_map(z, ath, rs; T=T) .- z
    sol = nlsolve(z -> G(z), z0; xtol=1e-8, ftol=1e-10, autodiff=:forward, method=:trust_region)
    return sol.zero
end

"""
    floquet_multipliers(zstar, ath, rs; T=7.0, ε=1e-6)

Finite-difference Jacobian of the Poincaré map DΦ_T at z*; returns eigenvalues (μ_i).
"""
function floquet_multipliers(zstar, ath, rs; T=7.0, ε=1e-6)
    n = length(zstar)
    Φz = poincare_map(zstar, ath, rs; T=T)
    M = Matrix{Float64}(undef, n, n)
    for j in 1:n
        ej = zeros(n); ej[j]=1.0
        Φz_pert = poincare_map(zstar .+ ε*ej, ath, rs; T=T)
        M[:,j] = (Φz_pert .- Φz) ./ ε
    end
    μ = eigvals(M)
    return μ, M
end

"""
    steady_weekly_summary(ath, rs; T=7.0)

Compute weekly fixed point, Floquet multipliers, and steady-week envelope of P(t).
"""
function steady_weekly_summary(ath, rs; T=7.0)
    zstar = poincare_fixed_point(ath, rs; T=T)
    μ, _ = floquet_multipliers(zstar, ath, rs; T=T)
    # Integrate one steady week for envelope
    sol = _integrate_interval(zstar, ath, rs; T=T, saveat=1/24)
    us = reduce(hcat, sol.u)'
    usT = permutedims(us, (2,1))
    P = readiness_series(sol.t, usT, ath.p)
    env = (Pmin = minimum(P), Pmed = median(P), Pmax = maximum(P))
    risky = count(>(0.8), usT[6,:]) / length(P)
    return (zstar=zstar, multipliers=μ, env=env, frac_time_I_high=risky, ts=sol.t, P=P, states=usT)
end
