using Pkg
# Pkg.add(["DifferentialEquations", "Dates"])  # run once

include("src/AthleteModel.jl")
using .AthleteModel

# Choose athletes
ath_nom = Nominal()
ath_SAE = SAE()
ath_SAS = SAS()
ath_SAR = SAR()

# Choose regimes
r1 = regime_early_HIIT()
r2 = regime_evening_intensity()
r3 = regime_split_day()
r4 = regime_alternating_hard_easy()
r5 = regime_midday()
r6 = regime_taper(weeks=3)
r7 = regime_sleep_extension()
r8 = regime_polarized()

# Simulate a few combinations (feel free to add/remove)
readouts = Dict{String,Readout}()

readouts["Nominal_EarlyHIIT"]   = simulate!(ath_nom, r1; T_days=42.0, saveat=1/24)
readouts["Nominal_Evening"]     = simulate!(ath_nom, r2; T_days=42.0, saveat=1/24)
readouts["SAE_Polarized"]       = simulate!(ath_SAE, r8; T_days=42.0, saveat=1/24)
readouts["SAS_Split"]           = simulate!(ath_SAS, r3; T_days=42.0, saveat=1/24)
readouts["SAR_AltHardEasy"]     = simulate!(ath_SAR, r4; T_days=42.0, saveat=1/24)
readouts["Nominal_Taper3wks"]   = simulate!(ath_nom, r6; T_days=21.0, saveat=1/24)
readouts["SAR_SleepExtension"]  = simulate!(ath_SAR, r7; T_days=28.0, saveat=1/24)

# Example: inspect a result
rd = readouts["Nominal_Taper3wks"]
println("Taper metrics: ", rd.metrics)

"""
    ascii_plot_P(rd::Readout; width=100, height=12)

Render a tiny ASCII chart of readiness P(t). Safe against off-by-one and rounding issues.
"""
function ascii_plot_P(rd::Readout; width::Int=100, height::Int=12)
    P = rd.P
    T = length(P)
    if T == 0
        println("(no data)"); return
    end

    # 1) do not sample more columns than points
    W = min(width, T)

    # 2) choose sample indices in [1, T], guaranteed inclusive of both ends
    raw  = range(1, T; length=W)
    idx  = clamp.(round.(Int, raw), 1, T)
    # ensure strictly nondecreasing and last == T
    idx[1] = 1
    idx[end] = T
    # if rounding produced duplicates, keep unique but preserve order
    idx = unique(idx)

    # Degenerate case: if only one unique index, draw a single marker
    if length(idx) < 2
        println("•"); return
    end
    W = length(idx)

    # Normalize P to [0,1] for plotting
    pmin, pmax = extrema(P)
    span = max(pmax - pmin, eps(Float64))
    normP = (P .- pmin) ./ span

    # Canvas
    canvas = fill(' ', height, W)
    # draw point markers (one per sampled column)
    for j in 1:W
        y = 1 + (height - 1) * (1 - normP[idx[j]])  # row (1=top)
        r = clamp(round(Int, y), 1, height)
        canvas[r, j] = '•'
    end
    # draw simple connections between consecutive samples
    for j in 1:(W-1)  # NOTE: W-1 (no look-ahead past end)
        r1 = findfirst(!=(' '), canvas[:, j])  |> x -> x === nothing ? height : x
        r2 = findfirst(!=(' '), canvas[:, j+1]) |> x -> x === nothing ? height : x
        if r1 == r2
            canvas[r1, j+1] = '─'
        elseif r1 < r2
            canvas[r1:(r2), j+1] .= '╲'
        else
            canvas[r2:(r1), j+1] .= '╱'
        end
    end

    # Print top→bottom
    for r in 1:height
        println(String(canvas[r, :]))
    end
end

ascii_plot_P(readouts["Nominal_EarlyHIIT"])
