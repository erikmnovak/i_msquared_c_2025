# Readiness P(t) and summary metrics

"""
    readiness(A,N,Fa,Fc,S,I,p)

Scalar readiness aggregator P(t) = w_end*A + w_str*N − λ··states.
"""
readiness(A,N,Fa,Fc,S,I,p::ModelParams) = p.w_end*A + p.w_str*N - p.λa*Fa - p.λc*Fc - p.λs*S - p.λi*I

"""
    readiness_series(ts, us, p)

Compute P(t) for a solved trajectory.
- `ts`: time vector
- `us`: state matrix (6×T)
"""
function readiness_series(ts, us, p::ModelParams)
    T = length(ts)
    P = Vector{Float64}(undef, T)
    for k in 1:T
        A,N,Fa,Fc,S,I = us[:,k]
        P[k] = readiness(A,N,Fa,Fc,S,I,p)
    end
    return P
end

"""
    summarize_metrics(ts, P, us; I_thresh=0.8)

Return a NamedTuple of simple decision metrics:
- max_P, t_to_peak, mean_weekly_P, frac_time_high_I
"""
function summarize_metrics(ts, P, us; I_thresh=0.8)
    # Max and time-to-peak
    idx = argmax(P)
    maxP = P[idx]
    t_peak = ts[idx]

    # Weekly mean (assumes ts in days and spans multiple weeks)
    horizon = ts[end] - ts[1]
    meanP_weekly = mean(P)  # already daily-resolved; fine for quick reporting

    # Risk time: proportion with I > threshold
    I = us[6,:]
    frac_risky = count(>(I_thresh), I) / length(I)

    return (max_P=maxP, t_to_peak=t_peak, mean_P=meanP_weekly, frac_time_I_high=frac_risky)
end

# lightweight Readout object for organized returns
Base.@kwdef struct Readout
    ts::Vector{Float64}
    states::Matrix{Float64}  # 6 × T
    P::Vector{Float64}
    metrics::NamedTuple
end
