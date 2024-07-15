using DataFrames
using LogExpFunctions

include("utils.jl")

function mixture_rl(data, pj::Vector{Float64}, r, cutoff::Float64, epsilon::Float64)

    function packparam(alpha::Matrix{Float64}, beta::Matrix{Float64}, lambda::Vector{Float64})
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda)
    end

    function calc_responsibilities(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = length(alpha)
        @assert Z == length(beta)
        @assert Z == length(lambda)

        players = groupby(data, :id)
        I = length(players)
        r = zeros(I, Z)
        log_lls = zeros(I, Z)
        
        for (i, player) in enumerate(players)
            payoffs = 0
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utility = alpha .+ delta_rd .* beta .+ lambda .* payoffs
                prob = sigmoid.(utility)
                log_lls[i,:] .+= (a>=cutoff) .* logfinite(epsilon).(prob) .+ (a<cutoff) .* logfinite(epsilon).(ones(Z) .- prob)
                payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
            end
        end
        
        for i in 1:I
            regs = logfinite(epsilon).(pj) .+ log_lls[i,:]
            for j in 1:Z
                r[i, j] = logfinite(epsilon)(pj[j]) + log_lls[i, j] - (maximum(regs) + logsumexp(regs .- maximum(regs)))
            end
        end
        r = exp.(r)
        return r
    end

    function calc_pj()
        pj = mean(eachrow(r))
        return pj
    end

    function expected_loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = length(alpha)
        @assert Z == length(beta)
        @assert Z == length(lambda)

        players = groupby(data, :id)
        
        exp_loglike = 0
        for (i, player) in enumerate(players)
            payoffs = 0
            Ti = nrow(player)
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utility = alpha .+ delta_rd .* beta .+ lambda .* payoffs
                prob = sigmoid.(utility)
                for j in 1:Z
                    exp_loglike += r[i,j] * (logfinite(epsilon)(pj[j]) / Ti + (a >= cutoff) * logfinite(epsilon)(prob[j]) + (a < cutoff) * logfinite(epsilon)(1-prob[j]))
                end
                payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
            end
        end
        return exp_loglike
    end

    function predict(theta)
        alpha, beta, lambda = unpackparam(theta)
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = 0
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                prob = sigmoid(utility)
                pred = ifelse(prob >= cutoff, 1, 0)
                push!(pred_df, (row.id, a, pred, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, dleta_rd, row.session, row.study))
                payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
            end
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda = unpackparam(theta)
        acc = 0
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = 0
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                prob = sigmoid(utility)
                acc += (a >= cutoff) * (prob >= cutoff) + (a < cutoff) * (prob < cutoff)
                payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
            end
        end
        return acc / nrow(data)
    end

    function loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = length(pj)
        @assert Z == length(alpha)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        loglike = 0
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            for j in 1:Z
                ll_j = 0
                payoffs = 0
                for row in eachrow(player)
                    a = row.a
                    delta_rd = row.delta_rd
                    utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                    prob = sigmoid(utility)
                    ll_j += (a >= cutoff) * logfinite(epsilon)(prob) + (a < cutoff) * logfinite(epsilon)(1 - prob)
                    payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
                end
                ll_j += logfinite(epsilon)(pj[j])
                loglike += logfinite(epsilon)(exp(ll_j))
            end
        end
        return loglike / nrow(data)
    end

    return (exp_ll = expected_loglike, pack = packparam, unpack = unpackparam, calc_r = calc_responsibilities, calc_pj = calc_pj, accuracy = accuracy, predict = predict, loglike = loglike)
end

function mixture_rl_mult(data, pj::Vector{Float64}, r, cutoffs::Vector{Float64}, epsilon::Float64)
    function packparam(alpha::Matrix{Float64}, beta::Matrix{Float64}, lambda::Vector{Float64})
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda)
    end

    function calc_responsibilities(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1
        players = groupby(data, :id)
        I = length(players)
        r = zeros(I, Z)
        log_lls = zeros(I, Z)
        
        for (i, player) in enumerate(players)
            payoffs = zeros(K)
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                ak = get_action(a, cutoffs)
                for j in 1:Z
                    utilities = alpha[j,:] .+ delta_rd .* beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                    probs = softmax_2(-utilities)
                    push!(probs, 1 - sum(probs))
                    log_lls[i,j] += logfinite(epsilon)(probs[ak])
                end
                payoffs[ak] += row.payoff
            end
        end
        
        for i in 1:I
            regs = logfinite(epsilon).(pj) .+ log_lls[i,:]
            for j in 1:Z
                r[i, j] = logfinite(epsilon)(pj[j]) + log_lls[i, j] - (maximum(regs) + logsumexp(regs .- maximum(regs)))
            end
        end
        r = exp.(r)
        return r
    end

    function calc_pj()
        pj = mean(eachrow(r))
        return pj
    end

    function expected_loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1

        players = groupby(data, :id)
        exp_loglike = 0
        for (i, player) in enumerate(players)
            for j in 1:Z
                payoffs = zeros(K)
                ll_j = 0
                for row in eachrow(player)
                    a = row.a
                    delta_rd = row.delta_rd
                    utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                    probs = softmax_2(-utilities)
                    push!(probs, 1 - sum(probs))
                    ak = get_action(a, cutoffs)
                    ll_j += logfinite(epsilon)(probs[ak])
                    payoffs[ak] += row.payoff
                end
                exp_loglike += r[i,j] * (logfinite(epsilon)(pj[j]) + ll_j)
            end
        end
        return exp_loglike
    end

    function predict(theta)
        alpha, beta, lambda = unpackparam(theta)
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = zeros(K)
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                probs = softmax_2(-utilities)
                push!(probs, 1 - sum(probs))
                max_prob, pred = findmax(probs)
                push!(pred_df, (row.id, a, pred - 1, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
                ak = get_action(a, cutoffs)
                payoffs[ak] += row.payoff
            end
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1
        acc = 0
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = zeros(K)
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                probs = softmax_2(-utilities)
                push!(probs, 1 - sum(probs))
                max_prob, pred = findmax(probs)
                ak = get_action(a, cutoffs)
                acc += (pred == ak)
                payoffs[ak] += row.payoff
            end
        end
        return acc / nrow(data)
    end

    function loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1
        
        loglike = 0
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            for j in 1:Z
                ll_j = 0
                payoffs = zeros(K)
                for row in eachrow(player)
                    a = row.a
                    delta_rd = row.delta_rd
                    utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                    probs = softmax_2(-utilities)
                    push!(probs, 1 - sum(probs))
                    ak = get_action(a, cutoffs)
                    ll_j += logfinite(epsilon)(probs[ak])
                    payoffs[ak] += row.payoff
                end
                ll_j += logfinite(epsilon)(pj[j])
                loglike += logfinite(epsilon)(exp(ll_j))
            end
        end
        return loglike / nrow(data)
    end

    return (exp_ll = expected_loglike, pack = packparam, unpack = unpackparam, calc_r = calc_responsibilities, calc_pj = calc_pj, accuracy = accuracy, predict = predict, loglike = loglike)
end
