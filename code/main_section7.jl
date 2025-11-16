include("src/AthleteModel.jl")
using .AthleteModel

# Athletes
aths = [Nominal(), SAE(), SAS(), SAR()]

# Regimes to analyze
regs = Dict(
  "EarlyHIIT"      => regime_early_HIIT(),
  "Evening"        => regime_evening_intensity(),
  "AltHardEasy"    => regime_alternating_hard_easy(),
  "Taper3w"        => regime_taper(weeks=3)
)

# 1) Stationary-input (equilibrium + Jacobian) for each combo
for ath in aths, (rname, rs) in regs
    eq = equilibrium_summary(ath, rs)
    js = jacobian_summary(eq.J)
    println("== $(ath.name) × $(rname) ==")
    println("Equilibrium z* = ", round.(eq.zstar; digits=3), "  P*=", round(eq.Pstar; digits=3))
    println("λ_max(real) = ", round(js.λmax; digits=4), "  half-life(days) ≈ ", round(js.halflife; digits=2))
end

# 2) Weekly periodic (Poincaré/Floquet) for selected regimes
sel = ["EarlyHIIT","Evening","AltHardEasy","Taper3w"]
for ath in aths, rname in sel
    rs = regs[rname]
    st = steady_weekly_summary(ath, rs)
    μmax = maximum(abs.(st.multipliers))
    println("== Weekly steady rhythm: $(ath.name) × $(rname) ==")
    println("|μ|_max = ", round(μmax; digits=4),
            "   P[min,med,max] = ", (round(st.env.Pmin;digits=3),
                                      round(st.env.Pmed;digits=3),
                                      round(st.env.Pmax;digits=3)),
            "   frac time I>0.8 = ", round(st.frac_time_I_high; digits=3))
end
