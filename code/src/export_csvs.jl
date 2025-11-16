# export_csvs.jl
# Create CSVs for PGFPlots: each file has two columns (t, P) with no header.
# Output directory: ./figdata/

# 1) Load your model ----------------------------------------------------------
include(joinpath(@__DIR__, "AthleteModel.jl"))   # adjusts for current folder
using .AthleteModel                               # exports simulate!, regimes, profiles, Readout
using DelimitedFiles                              # for writedlm

# 2) Helpers ------------------------------------------------------------------
const OUTDIR = joinpath(@__DIR__, "figdata")
isdir(OUTDIR) || mkpath(OUTDIR)

"""
    save_readiness_csv(ro::Readout, path::AbstractString)

Write two columns (time in days, readiness P) with comma separators and no header.
Compatible with `\addplot table[x index=0, y index=1, col sep=comma]{...}` in PGFPlots.
"""
function save_readiness_csv(ro::Readout, path::AbstractString)
    data = [ro.ts ro.P]           # T × 2
    writedlm(path, data, ',')
    println("SAVED  $(basename(path))  →  $(path)  (rows=$(length(ro.ts)))")
end

"""
    run_and_save(ath, rs; T_days, name)

Simulate a regime `rs` for athlete profile `ath` over `T_days` days and save readiness CSV `name`.
Uses a 1/24 day save step (hourly) to match your LaTeX figure.
"""
function run_and_save(ath::AthleteProfile, rs::RegimeSpec; T_days::Float64, name::AbstractString)
    ro = simulate!(ath, rs; T_days=T_days, saveat=1/24)   # uses your ODE + readiness map
    save_readiness_csv(ro, joinpath(OUTDIR, name))
    return ro
end

# 3) Scenarios (exactly the ones used in the figure) --------------------------
# Athletes
ath_nom = Nominal()
ath_SAE = SAE()
ath_SAS = SAS()
ath_SAR = SAR()

# Regimes
reg_early    = regime_early_HIIT()               # Early‑AM HIIT
reg_evening  = regime_evening_intensity()        # Evening intensity
reg_polar    = regime_polarized()                # Polarized (midday)
reg_split    = regime_split_day()                # Split day (AM E + PM S)
reg_altEZ    = regime_alternating_hard_easy()    # Alternating hard/easy
reg_taper    = regime_taper()                    # Taper (volume↓, intensity kept, early)
reg_sleepExt = regime_sleep_extension()          # Sleep extension + nap

# 4) Run + export -------------------------------------------------------------
# 42‑day training blocks
run_and_save(ath_nom, reg_early;   T_days=42.0, name="nominal_early_am_hiit.csv")
run_and_save(ath_nom, reg_evening; T_days=42.0, name="nominal_evening_intensity.csv")
run_and_save(ath_SAE, reg_polar;   T_days=42.0, name="sae_polarized_midday.csv")
run_and_save(ath_SAS, reg_split;   T_days=42.0, name="sas_split_day.csv")
run_and_save(ath_SAR, reg_altEZ;   T_days=42.0, name="sar_alt_hard_easy.csv")

# shorter horizons by design (these curves end earlier on the shared x‑axis)
run_and_save(ath_nom, reg_taper;    T_days=21.0, name="nominal_taper.csv")
run_and_save(ath_SAR, reg_sleepExt; T_days=28.0, name="sar_sleep_extension.csv")

println("\nAll readiness CSVs written to: $(OUTDIR)")
