# Integrators + progress prints

using DifferentialEquations

"""
    simulate!(ath::AthleteProfile, rs::RegimeSpec; T_days=42.0, saveat=1/24)

Integrate the six-state ODE for `T_days` under `rs` using parameters from `ath`.
Returns a `Readout` with time `ts`, states, readiness `P`, and summary metrics.

Monitors progress with simple INFO prints.
"""
function simulate!(ath::AthleteProfile, rs::RegimeSpec; T_days=42.0, saveat=1/24)
    p = ath.p
    println("────────────────────────────────────────────────────────────")
    println("INFO  Athlete: $(ath.name)")
    println("INFO  Regime : $(rs.name)")
    println("INFO  Desc   : $(rs.description)")
    println("INFO  Horizon: $(T_days) days, output step = $(saveat) day")
    println("INFO  Building B(t) and q(t) …")

    # Build B(t) and q(t) using the regime and parameters
    B = B_of_t_factory(rs.uE, rs.uH, rs.uS, p, rs)
    q = q_of_B_factory(p)(B)

    # bundle inputs for rhs!
    inputs = (uE=rs.uE, uH=rs.uH, uS=rs.uS, s=rs.s, n=rs.n, x=rs.x, B=B, q=q)

    # Initial state
    u0 = [p.A0, p.N0, p.Fa0, p.Fc0, p.S0, p.I0]

    # Problem and solver options
    prob = ODEProblem((du,u,pp,t)->rhs!(du,u,p,t,inputs), u0, (0.0, T_days))
    println("INFO  Solving ODE (RKDP5) with reltol=1e-6, abstol=1e-6 …")
    sol = solve(prob, DP5(); saveat=saveat, reltol=1e-6, abstol=1e-6, maxiters=1_000_000)

    println("INFO  Done. Number of saved steps = $(length(sol.t))")

    # Build readiness series and metrics
    us = reduce(hcat, sol.u)'  # T × 6; transpose next line
    usT = permutedims(us, (2,1)) # 6 × T
    P  = readiness_series(sol.t, usT, p)
    metrics = summarize_metrics(sol.t, P, usT; I_thresh=0.8)

    println("INFO  Peak P = $(round(metrics.max_P, digits=3)) at day $(round(metrics.t_to_peak,digits=2))")
    println("INFO  Mean P (overall) = $(round(metrics.mean_P, digits=3))")
    println("INFO  Fraction of time with I > 0.8 = $(round(metrics.frac_time_I_high,digits=3))")
    println("────────────────────────────────────────────────────────────")

    return Readout(ts=sol.t, states=usT, P=P, metrics=metrics)
end
