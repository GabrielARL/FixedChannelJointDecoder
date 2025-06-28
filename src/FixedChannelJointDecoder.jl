module FixedChannelJointDecoder

using LinearAlgebra
using Random
using Optim
using LineSearches
using SparseArrays
using GaloisFields

const GF2 = GaloisField(2)

# === Systematic LDPC Construction ===

export generate_systematic_ldpc, encode_ldpc_systematic, is_valid_codeword
export decode_unbounded, decode_unbounded_until_valid
export loss_unbounded, grad_unbounded!
export decode_soft_until_valid
"""
    generate_systematic_ldpc(k, m, d_c)

Constructs a systematic LDPC parity-check matrix H = [H_m | I], where H_p = I.
Returns a sparse matrix H of size m × (k + m).
"""
function generate_systematic_ldpc(k::Int, m::Int, d_c::Int)
    n = k + m
    Hm = falses(m, k)
    for col in 1:k
        rows = randperm(m)[1:d_c]
        Hm[rows, col] .= true
    end
    Hp = Matrix(I, m, m)  # Identity matrix
    return sparse(hcat(Hm, Hp))
end

"""
    encode_ldpc_systematic(H, msg)

Encodes a message using a systematic parity-check matrix H = [H_m | H_p].
Solves for parity bits using GF(2) Gaussian elimination.
"""
function encode_ldpc_systematic(H::SparseMatrixCSC{Bool}, msg::Vector{Bool})
    m, n = size(H)
    k = n - m
    Hm = Array(H[:, 1:k])
    Hp = Array(H[:, k+1:end])
    rhs = -GF2.(Hm) * GF2.(msg)
    A = hcat(GF2.(Hp), rhs)

    for i in 1:m
        if A[i, i] == GF2(0)
            for j in i+1:m
                if A[j, i] != GF2(0)
                    A[i, :], A[j, :] = A[j, :], A[i, :]
                    break
                end
            end
        end
        for j in 1:m
            if i != j && A[j, i] != GF2(0)
                A[j, :] += A[i, :]
            end
        end
    end
    p = A[:, end]
    return Int.(vcat(msg, map(x -> x == GF2(1), p)))
end

"""
    is_valid_codeword(bits, H)

Checks whether the input vector `bits` satisfies the parity-check matrix `H`.
"""
function is_valid_codeword(bits::Vector{Int}, H::SparseMatrixCSC)
    x = GF2.(bits)
    return all(H * x .== GF2(0))
end

# === Decoder Utilities ===

"""
    myconv(x, h)

Simple causal convolution: (x * h)[1:length(x)]
"""
function myconv(x::Vector{<:Real}, h::Vector{<:Real})
    n, L = length(x), length(h)
    return [sum(h[j] * x[i - j + 1] for j in 1:L if 1 <= i - j + 1 <= n) for i in 1:(n + L - 1)]
end

"""
    loss_unbounded(z, y, h, parity_indices, λ, γ)

Decoding loss = data term + parity constraint + regularization
"""
function loss_unbounded(z, y, h, parity_indices, λ, γ)
    n = length(z)
    x_bpsk = tanh.(z)
    ŷ = myconv(x_bpsk, h)[1:n]
    data_term = sum((ŷ .- y).^2)
    constraint_term = sum((1 - prod(x_bpsk[inds]))^2 for inds in parity_indices)
    reg_term = γ * sum(z .^ 2)
    return data_term + λ * constraint_term + reg_term
end

"""
    grad_unbounded!(g, z, y, h, parity_indices, λ, γ)

Gradient of `loss_unbounded` w.r.t z. Safe and efficient.
"""
function grad_unbounded!(g, z, y, h, parity_indices, λ, γ)
    n = length(z)
    x_bpsk = tanh.(z)
    ŷ = myconv(x_bpsk, h)[1:n]
    res = 2 .* (ŷ .- y)

    ∂L_∂x = zeros(n)
    for i in 1:n
        for j in 1:length(h)
            k = i - j + 1
            if 1 <= k <= n
                ∂L_∂x[k] += res[i] * h[j]
            end
        end
    end

    for inds in parity_indices
        t = x_bpsk[inds]
        total_prod = prod(t)
        t_safe = clamp.(t, -0.999, 0.999)
        scale = -2λ * (1 - total_prod)
        for (j, idx) in enumerate(inds)
            ∂prod_∂xj = total_prod / t_safe[j]
            ∂L_∂x[idx] += scale * ∂prod_∂xj
        end
    end

    sech2 = 1 .- x_bpsk.^2
    g[:] = ∂L_∂x .* sech2 .+ 2γ .* z
    return g
end

# === Decoding API ===

"""
    decode_unbounded(H, parity_indices, y, h; λ, γ, max_attempts)

Gradient-based LDPC decoder using iterative retry strategy.
"""
function decode_unbounded(H, parity_indices, y, h; λ=0.25, γ=1e-3, max_attempts=10)
    n = length(y)
    power = sum(abs2, h)
    z0 = 2 .* y ./ (power + 1e-8)

    obj = z -> loss_unbounded(z, y, h, parity_indices, λ, γ)
    grad! = (g, z) -> grad_unbounded!(g, z, y, h, parity_indices, λ, γ)

    od = Optim.OnceDifferentiable(obj, grad!, z0)
    result = optimize(od, z0, LBFGS(), Optim.Options(; iterations=100))
    ẑ = Optim.minimizer(result)
    x_bpsk = tanh.(ẑ)

    for attempt in 1:max_attempts
        x_hard = @. Int(x_bpsk < 0)
        if is_valid_codeword(x_hard, H)
            return x_bpsk, true, attempt
        end
        ẑ = ẑ .+ 0.01 * randn(n)
        x_bpsk = tanh.(ẑ)
    end

    return x_bpsk, false, max_attempts
end

"""
    decode_unbounded_until_valid(H, parity_indices, y, h; λ, γ, max_attempts, ε)

Retry decoder that perturbs latent vector z and retries.
"""
function decode_unbounded_until_valid(H, parity_indices, y, h;
    λ=0.25, γ=1e-3, max_attempts=5, ε=0.01)

    n = length(y)
    power = sum(abs2, h)
    z = 2 .* y ./ (power + 1e-8)

    obj = z -> loss_unbounded(z, y, h, parity_indices, λ, γ)
    grad! = (g, z) -> grad_unbounded!(g, z, y, h, parity_indices, λ, γ)

    for attempt in 1:max_attempts
        od = Optim.OnceDifferentiable(obj, grad!, z)
        result = optimize(od, z, LBFGS(), Optim.Options(; iterations=100))
        z = Optim.minimizer(result)
        x_bpsk = tanh.(z)
        x_hard = @. Int(x_bpsk < 0)

        if is_valid_codeword(x_hard, H)
            return x_bpsk, true, attempt
        end

        z .= z .+ ε * randn(n)
    end

    x_bpsk = tanh.(z)
    return x_bpsk, false, max_attempts
end

function decode_soft_until_valid(x_bpsk_init::Vector{Float64}, σ::Float64, parity_indices::Vector{Vector{Int}}, H::SparseMatrixCSC{Bool}; 
                                 max_attempts::Int=5, ε::Float64=0.01)
    n = length(x_bpsk_init)
    x_bpsk = copy(x_bpsk_init)
    damping = 0.5
    success = false
    attempt = 0
    prev_x = copy(x_bpsk)

    while attempt < max_attempts
        for inds in parity_indices
            p = prod(x_bpsk[inds])
            for i in inds
                x_bpsk[i] += damping * (p / x_bpsk[i] - x_bpsk[i])
            end
        end

        if norm(x_bpsk .- prev_x) < ε
            break
        end

        prev_x .= x_bpsk
        attempt += 1
    end

    x_hard = @. Int(x_bpsk < 0)
    success = all(H * GaloisField(2).(x_hard) .== 0)

    return x_bpsk, x_hard, attempt, success, norm(x_bpsk .- prev_x)
end


end # module LDPCDecoder
