# Regime builders and input streams (uE,uH,uS,s,n,x), bedtime kernel B(t), sleep efficiency q(B)

using Dates

# ------------------------------ Types ------------------------------

"""
RegimeSpec bundles callable inputs:
uE(t), uH(t), uS(t), s(t), n(t), x(t), and bedtime kernel B(t) + q(B).
It also stores daily timing: bedtime hour and sleep duration, plus optional naps.
"""
Base.@kwdef struct RegimeSpec
    name::String
    uE::Function
    uH::Function
    uS::Function
    s::Function
    n::Function
    x::Function
    bedtime_hour::Float64 = 23.0     # 23:00 by default
    sleep_hours::Float64  = 8.0
    naps::Vector{Tuple{Float64,Float64}} = Vector{Tuple{Float64,Float64}}() # (start_hour, dur_hours)
    # For prints/metadata
    description::String = ""
end

# ------------------------------ Utilities ------------------------------
const Nap = Tuple{Float64,Float64}   # (start_hour, duration_hours) for naps

function s_of_t_factory(rs::RegimeSpec)
    # Night sleep (wrap across midnight)
    bed = rs.bedtime_hour / 24.0
    dur = rs.sleep_hours   / 24.0
    naps = rs.naps::Vector{Nap}      # ensure (hour, duration) pairs

    function s(t::Real)
        τ  = t - floor(t)            # time-of-day in days
        # inside nightly block: from (bed - dur) to bed, wrapping at 0
        in_night = (τ >= bed - dur) || (τ < bed)

        # any naps? check each (start,duration) pair; convert each component to days
        in_nap = false
        @inbounds for (h, d) in naps
            hs = h / 24.0
            he = (h + d) / 24.0
            in_nap |= (τ >= hs) & (τ < he)
        end

        return (in_night || in_nap) ? 1.0 : 0.0   # return a scalar, not Bool
    end
    return s
end

# A training block is (start_hour, duration_hours, amplitude)
const Block = Tuple{Float64,Float64,Float64}
const Nap   = Tuple{Float64,Float64}   # (start_hour, duration_hours)

# Convert hour-based blocks to day-based blocks without tuple×scalar
to_day_blocks(blocks::Vector{Block}) =
    [(h/24.0, d/24.0, a) for (h,d,a) in blocks]

to_day_naps(naps::Vector{Nap}) =
    [(h/24.0, d/24.0) for (h,d) in naps]


# Convert hours → days (our integrator runs in days)
const H2D = 1.0/24.0

# Rectangular pulse centered at hour h0 with duration dur_h, amplitude amp
# Time 't' is in days; we tile pulses every day according to a schedule.
function rect_pulse_daily(t::Float64; h0::Float64, dur_h::Float64, amp::Float64)
    t_in_day = t - floor(t)                     # [0,1)
    start_d  = h0*H2D
    stop_d   = (h0 + dur_h)*H2D
    return (start_d ≤ t_in_day < stop_d) ? amp : 0.0
end

# Multiple pulses per day (sum)
function multi_pulse_daily(t; sessions::Vector{Tuple{Float64,Float64,Float64}})
    # sessions: [(hour, dur_h, amp), ...]
    s = 0.0
    @inbounds for (h, d, a) in sessions
        s += rect_pulse_daily(t; h0=h, dur_h=d, amp=a)
    end
    return s
end

# Sleep schedule s(t): nightly block + optional naps
function sleep_indicator(t; bedtime_hour::Float64=23.0, sleep_hours::Float64=8.0, naps=Tuple{Float64,Float64}[])
    t_in_day = t - floor(t)
    start_d  = bedtime_hour*H2D
    stop_d   = (bedtime_hour + sleep_hours)*H2D
    base = (t_in_day ≥ start_d || t_in_day < stop_d) ? 1.0 : 0.0
    if !isempty(naps)
        for (h, dur) in naps
            base = (base == 1.0 || (rect_pulse_daily(t; h0=h, dur_h=dur, amp=1.0) > 0.0)) ? 1.0 : 0.0
        end
    end
    return base
end

# Context stress default: small constant (can add spikes/travel days)
x_const(c::Float64) = t -> c

# Nutrition default: moderate, constant, with optional post-session bumps could be added
n_const(c::Float64) = t -> c


# Provide defaults and a keyword constructor via @kwdef
Base.@kwdef struct RegimeSpec
    name::String
    uE::Function         # endurance input uE(t)
    uH::Function         # HIIT/anaerobic input uH(t)
    uS::Function         # strength/plyo input uS(t)
    s::Function          # sleep-opportunity indicator s(t)
    n::Function          # nutrition/availability n(t)
    x::Function          # context stress x(t)
    bedtime_hour::Float64 = 22.0
    sleep_hours::Float64 = 8.0
    naps::Vector{Nap} = Nap[]          # optional naps within the day
    description::String = ""
end

# Explicit wrapper that forwards keywords to the positional ctor
RegimeSpec(name::AbstractString, uE::Function, uH::Function, uS::Function,
           s::Function, n::Function, x::Function;
           bedtime_hour::Real=22.0, sleep_hours::Real=8.0,
           naps::Vector{Nap}=Nap[], description::AbstractString="") =
    RegimeSpec(String(name), uE, uH, uS, s, n, x,
               float(bedtime_hour), float(sleep_hours), naps, String(description))

# ------------------------------ Bedtime kernel B and sleep efficiency q ------------------------------

"""
Compute bedtime kernel B(t) from the *past hb* hours relative to "lights out".

We approximate B(t) analytically by summing decayed contributions from today's pulses that
occur in the last hb hours before bedtime. For simplicity we only look at "today's" pulses; this
is consistent since hb ≤ 6 h.
"""
function B_of_t_factory(uE::Function, uH::Function, uS::Function,
                        p::ModelParams, rs)
    # Pull scalars (do NOT pack into a tuple and multiply later)
    βE, βH, βS = p.βE, p.βH, p.βS

    # Convert hours → days once
    σ  = p.σb / 24.0
    H  = p.hb  / 24.0

    # Simple Riemann rule over the last H days (i.e., last hb hours)
    n = 24                        # 24 slices over the horizon
    Δ = H / n

    # Exponential memory on [0, H]
    K(δ) = exp(-δ / σ)

    function B(t::Real)
        acc = 0.0
        @inbounds for i in 1:n
            δ = i * Δ
            acc += (βE * uE(t - δ) + βH * uH(t - δ) + βS * uS(t - δ)) * K(δ)
        end
        return acc * Δ
    end
    return B
end

q_of_B_factory(p::ModelParams) = (B::Function) -> (t -> p.q0 / (1.0 + p.η * B(t)))

# ------------------------------ Regime builders ------------------------------

# 1) Early‑morning HIIT (07:00–08:00), no nap
function regime_early_HIIT(; ampH=1.0, ampE=0.2, weeks=6, bedtime_hour=23.0, sleep_hours=8.0)
    uH_sessions = [(7.0, 1.0, ampH)]                   # hour, duration, amplitude
    uE_sessions = [(9.0, 0.5, ampE)]
    uS_sessions = Tuple{Float64,Float64,Float64}[]
    uE = t -> multi_pulse_daily(t; sessions = uE_sessions)
    uH = t -> multi_pulse_daily(t; sessions = uH_sessions)
    uS = t -> multi_pulse_daily(t; sessions = uS_sessions)
    s  = t -> sleep_indicator(t; bedtime_hour=bedtime_hour, sleep_hours=sleep_hours)
    n  = n_const(0.8)
    x  = x_const(0.1)
    RegimeSpec("Early‑AM HIIT", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="Short HIIT in the morning + small E, low B(t), no naps.")
end

# 2) Evening moderate/high intensity (19:00–21:00)
function regime_evening_intensity(; ampE=0.6, ampH=0.6, weeks=6, bedtime_hour=23.0, sleep_hours=8.0)
    uE_sessions = [(19.0, 1.0, ampE)]
    uH_sessions = [(20.0, 1.0, ampH)]
    uS_sessions = Tuple{Float64,Float64,Float64}[]
    uE = t -> multi_pulse_daily(t; sessions = uE_sessions)
    uH = t -> multi_pulse_daily(t; sessions = uH_sessions)
    uS = t -> multi_pulse_daily(t; sessions = uS_sessions)
    s  = t -> sleep_indicator(t; bedtime_hour=bedtime_hour, sleep_hours=sleep_hours)
    n  = n_const(0.8)
    x  = x_const(0.1)
    RegimeSpec("Evening intensity", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="E+H in the evening → large B(t), reduced q(B) that night.")
end

# 3) Split day (light AM endurance + PM strength)
function regime_split_day(; ampE=0.3, ampS=0.8, weeks=6, bedtime_hour=23.0, sleep_hours=8.0)
    uE_sessions = [(9.0, 0.75, ampE)]
    uH_sessions = Tuple{Float64,Float64,Float64}[]
    uS_sessions = [(18.0, 1.0, ampS)]
    uE = t -> multi_pulse_daily(t; sessions = uE_sessions)
    uH = t -> multi_pulse_daily(t; sessions = uH_sessions)
    uS = t -> multi_pulse_daily(t; sessions = uS_sessions)
    s  = t -> sleep_indicator(t; bedtime_hour=bedtime_hour, sleep_hours=sleep_hours)
    n  = n_const(0.85)
    x  = x_const(0.1)
    RegimeSpec("Split day (AM E + PM S)", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="AM endurance + PM strength (some B penalty).")
end

# 4) Alternating hard/easy microcycle (hard days: large pulses; easy days: minimal + nap)
function regime_alternating_hard_easy(; ampE=0.6, ampH=0.6, ampS=0.6, nap=(13.0,0.5), bedtime_hour=23.0, sleep_hours=8.0)
    # Hard day pulses; easy days: 10% of pulses.
    hard = [(10.0, 1.0, ampE), (17.0, 1.0, ampH), (18.0, 0.75, ampS)]
    easy = [(10.0, 0.3, 0.1*ampE)]
    function uE(t)
        day = Int(floor(t))
        sessions = iseven(day) ? hard : easy
        return multi_pulse_daily(t; sessions=[(h,d,a) for (h,d,a) in sessions if (h==10.0)])
    end
    function uH(t)
        day = Int(floor(t))
        sessions = iseven(day) ? [(17.0, 1.0, ampH)] : Tuple{Float64,Float64,Float64}[]
        return multi_pulse_daily(t; sessions=sessions)
    end
    function uS(t)
        day = Int(floor(t))
        sessions = iseven(day) ? [(18.0,0.75,ampS)] : Tuple{Float64,Float64,Float64}[]
        return multi_pulse_daily(t; sessions=sessions)
    end
    naps = [nap]
    s = t -> sleep_indicator(t; bedtime_hour, sleep_hours, naps)
    n = n_const(0.85)
    x = x_const(0.1)
    RegimeSpec("Alternating hard/easy", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours, naps,
               description="Hard days with E+H+S, easy days minimal + 30min nap.")
end

# 5) Midday polarized endurance (large E, tiny H pulled to midday)
function regime_midday(; ampE=0.8, ampH=0.2, weeks=6, bedtime_hour=23.0, sleep_hours=8.0)
    uE_sessions = [(13.0, 1.5, ampE)]
    uH_sessions = [(12.0, 0.5, ampH)]
    uS_sessions = Tuple{Float64,Float64,Float64}[]
    uE = t -> multi_pulse_daily(t; sessions=uE_sessions)
    uH = t -> multi_pulse_daily(t; sessions=uH_sessions)
    uS = t -> multi_pulse_daily(t; sessions=uS_sessions)
    s  = t -> sleep_indicator(t; bedtime_hour, sleep_hours)
    n  = n_const(0.8)
    x  = x_const(0.1)
    RegimeSpec("Midday polarized endurance", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="High E + tiny H placed far from bedtime, keeping B small.")
end

# 6) Taper (volume down, keep intensity, sessions earlier in the day)
function regime_taper(; weeks=3, base_ampE=0.6, base_ampH=0.6, bedtime_hour=22.0, sleep_hours=8.5)
    # Exponential volume decay across days; maintain brief H; early hours to keep B low
    function scaler(t)
        d = t
        return 0.6^((d)/7)   # halve every ~week
    end
    uE = t -> scaler(t) * multi_pulse_daily(t; sessions=[(9.0, 1.0, base_ampE)])
    uH = t -> multi_pulse_daily(t; sessions=[(10.0, 0.75, base_ampH)]) * min(1.0, 0.4 + 0.6*scaler(t))
    uS = t -> 0.0
    s  = t -> sleep_indicator(t; bedtime_hour, sleep_hours)
    n  = n_const(0.9)
    x  = x_const(0.05)
    RegimeSpec("Taper (vol ↓, intensity kept, early sessions)", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="Exponential volume reduction, intensity maintained, sessions early.")
end

# 7) Sleep‑extension with nap policy
function regime_sleep_extension(; base_ampE=0.5, bedtime_hour=22.5, sleep_hours=9.0, nap=(13.0,0.5))
    uE = t -> multi_pulse_daily(t; sessions=[(13.0, 1.0, base_ampE)])
    uH = t -> 0.0
    uS = t -> 0.0
    naps = [nap]
    s  = t -> sleep_indicator(t; bedtime_hour, sleep_hours, naps)
    n  = n_const(0.9)
    x  = x_const(0.05)
    RegimeSpec("Sleep extension + nap", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours, naps,
               description="Longer nightly sleep + short nap; moderate midday E.")
end

# 8) Polarized vs. pyramidal selector (here: polarized)
function regime_polarized(; ampE=0.9, ampH=0.2, bedtime_hour=23.0, sleep_hours=8.0)
    uE = t -> multi_pulse_daily(t; sessions=[(12.5, 1.5, ampE)])
    uH = t -> multi_pulse_daily(t; sessions=[(11.0, 0.5, ampH)])
    uS = t -> 0.0
    s  = t -> sleep_indicator(t; bedtime_hour, sleep_hours)
    n  = n_const(0.8)
    x  = x_const(0.08)
    RegimeSpec("Polarized (high E, little H, midday)", uE,uH,uS,s,n,x; bedtime_hour, sleep_hours,
               description="High E, small H, both midday → small B.")
end
