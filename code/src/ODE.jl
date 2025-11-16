# ODE right-hand side and gates

using LinearAlgebra

# Gates
Grec(S,Fc,p::ModelParams) = 1.0 / (1.0 + p.cS*S + p.cC*Fc)
Gint(uE,p::ModelParams)   = 1.0 / (1.0 + p.μE*uE)

# RHS in-place: du = f(u,p,t). State order: [A,N,Fa,Fc,S,I]
"""
    rhs!(du, u, p, t, inputs)

Compute the right-hand side of the six-state ODE.

Arguments
---------
- `du`: output derivative vector (modified in-place)
- `u`:  current state vector  (A, N, Fa, Fc, S, I)
- `p`:  `ModelParams`
- `t`:  time in days (Float64)
- `inputs`: NamedTuple with fields `uE,uH,uS,s,n,x,B,q` (all are callables B(t), q(t), etc.)

Notes
-----
- Sleep-efficiency `q(t)` multiplies nightly clearance terms via s(t)*q(t).
- Bedtime kernel `B(t)` is used *inside* q(t); it is provided separately for transparency/plots.
"""
function rhs!(du,u,p::ModelParams,t,inputs)
    A,N,Fa,Fc,S,I = u
    uE = inputs.uE(t); uH = inputs.uH(t); uS = inputs.uS(t)
    s  = inputs.s(t);  n  = inputs.n(t);  x  = inputs.x(t)
    q  = inputs.q(t)   # already computed from B(t) inside Inputs.jl

    # --- Top row: capacities
    dA = p.kA*(p.αE*uE + p.αH*uH)*Grec(S,Fc,p)*(1.0 - A/p.KA) - A/p.τA - p.θAI*I
    dN = p.kN*(p.αS*uS + p.αHN*uH)*Grec(S,Fc,p)*Gint(uE,p)*(1.0 - N/p.KN) - N/p.τN - p.θNI*I

    # --- Middle row: fatigue
    dFa = (p.γE*uE + p.γH*uH + p.γS*uS) - (1.0/p.τfa + p.ρa*s*q)*Fa
    dFc = p.ε*Fa + p.ξc*x - (1.0/p.τfc + p.ρc*s*q)*Fc

    # --- Bottom row: sleep debt / micro-damage
    dS  = p.λw*(1.0 - s) + p.λT*(p.γE*uE + p.γH*uH + p.γS*uS) + p.ξs*x - (1.0/p.τS + p.μs*s*q)*S
    load_to_I = p.ψ0*(p.κE*uE + p.κH*uH + p.κS*uS)*(1.0 + p.χa*Fa + p.χc*Fc)
    dI  = load_to_I - (1.0/p.τI + p.ψs*s*q + p.ψn*n)*I

    du[1]=dA; du[2]=dN; du[3]=dFa; du[4]=dFc; du[5]=dS; du[6]=dI
    return nothing
end
