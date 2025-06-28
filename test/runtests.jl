# test/runtests.jl

using Random, Printf, LinearAlgebra, SparseArrays, Statistics, Dates

#incFixedChannelJointDecoder.jl")  # use relative path

using FixedChannelJointDecoder

function run_ber_sweep_at_snr(snr_db::Float64; 
    num_trials::Int = 3, 
    k::Int = 1024, 
    m::Int = 512, 
    d_c::Int = 4,
    h::Vector{Float64} = [0.407, 0.815, 0.407],
    λ_unbounded::Float64 = 0.25,
    λ_bounded::Float64 = 0.25,
    γ::Float64 = 1e-3,
    max_attempts::Int = 2,
    ε::Float64 = 0.01)

    n = k + m
    σ = 10.0^(-snr_db / 20)
    ber1_total = 0.0
    ber2_total = 0.0
    valid1_total = 0
    valid2_total = 0
    time_unbounded_total = 0.0
    time_soft_total = 0.0

    H = FixedChannelJointDecoder.generate_systematic_ldpc(k, m, d_c)
    parity_indices = [findall(!iszero, H[i, :]) for i in 1:m]

    for trial in 1:num_trials
        msg = rand(Bool, k)
        x_true = FixedChannelJointDecoder.encode_ldpc_systematic(H, msg)
        x_bpsk = 1 .- 2 .* x_true
        y_clean = FixedChannelJointDecoder.myconv(x_bpsk, h)[1:n]
        y_noisy = y_clean .+ σ * randn(n)

        time1 = @elapsed begin
            x̂_bpsk, success1, _ = FixedChannelJointDecoder.decode_unbounded_until_valid(
                H, parity_indices, y_noisy, h;
                λ=λ_unbounded, γ=γ, max_attempts=max_attempts, ε=ε
            )
        end
        x̂_hard1 = @. Int(x̂_bpsk < 0)
        error_indices1 = findall(x̂_hard1 .!= x_true)
        ber1_total += length(error_indices1) / n
        valid1_total += success1
        time_unbounded_total += time1

        time2 = @elapsed begin
            x̂_bpsk2, _, _, success2, _ = FixedChannelJointDecoder.decode_soft_until_valid(
                x̂_bpsk, σ, parity_indices, H;
                max_attempts=max_attempts, ε=ε
            )
        end
        x̂_hard2 = @. Int(x̂_bpsk2 < 0)
        error_indices2 = findall(x̂_hard2 .!= x_true)
        ber2_total += length(error_indices2) / n
        valid2_total += success2
        time_soft_total += time2
    end

    ber1_avg = ber1_total / num_trials
    ber2_avg = ber2_total / num_trials
    success1_rate = valid1_total / num_trials
    success2_rate = valid2_total / num_trials
    time1_avg = time_unbounded_total / num_trials
    time2_avg = time_soft_total / num_trials

    println("\n📶 SNR = $(snr_db) dB → σ = $(round(σ, sigdigits=3))")
    @printf("🔁 Unbounded Joint Decode:     BER = %.6f | Success Rate = %.2f | Avg Time = %.2f ms\n", ber1_avg, success1_rate, time1_avg * 1000)
    @printf("🎯 Soft Refinement Decode:     BER = %.6f | Success Rate = %.2f | Avg Time = %.2f ms\n", ber2_avg, success2_rate, time2_avg * 1000)
end

# Example usage:
run_ber_sweep_at_snr(8.0; num_trials=3)
