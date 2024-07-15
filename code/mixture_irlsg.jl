using DataFrames
using LogExpFunctions

include("utils.jl")

function mixture_irlsg(data, pj::Vector{Float64}, r, cutoff::Float64, state_cutoff::Vector{Float64}, epsilon::Float64)
    function packparam(alpha::Vector{Float64}, beta::Vector{Float64}, lambda::Vector{Float64})
        S = length(state_cutoff) + 1
        Z = length(alpha)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        sigma = zeros(Z, S)
        data_non_init = filter(row ->row.period != 1, data)
        players = groupby(data_non_init, :id)
        I = length(players)
        n = zeros(I, S, 2)
        calc_responsibilities(ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma))
        # count for n matrix
        for (i, player) in enumerate(players)
            for row in eachrow(player)
                a = row.a
                state = get_state(row.prev_avg_a, state_cutoff)
                if a < cutoff
                    n[i, state, 1] += 1
                else
                    n[i, state, 2] += 1
                end
            end
        end
        # calculate sigmas
        for j in 1:Z
            for s in 1:S
                num = 0
                den = 0
                for i in 1:I
                    num += r[i, j] * n[i, s, 2]
                    den += r[i, j] * sum(n[i, s, :])
                    @assert r[i, j] * (n[i, s, 1] + n[i, s, 2]) == r[i, j] * sum(n[i, s, :])
                end
                # Laplace Smoothing
                sigma[j, s] = (num + 1) / (den + 2)
            end
        end
        
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda, theta.sigma)
    end

    function calc_responsibilities(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        Z = length(alpha)
        S = size(sigma, 2)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        @assert Z == size(sigma, 1)
        @assert S == length(state_cutoff) + 1

        players = groupby(data, :id)
        I = length(players)
        r = zeros(I, Z)
        log_lls = zeros(I, Z)
        for (i, player) in enumerate(players)
            for j in 1:Z
                payoffs = 0
                for row in eachrow(player)
                    period = row.period
                    a = row.a
                    delta_rd = row.delta_rd
                    if period != 1
                        state = get_state(row.prev_avg_a, state_cutoff)
                        prob = sigma[j, state]
                    else
                        payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                        utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                        prob = sigmoid(utility)
                    end
                    log_lls[i,j] += (a>=cutoff) * logfinite(epsilon)(prob) + (a<cutoff) * logfinite(epsilon)(1 - prob)
                end
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
        alpha, beta, lambda, sigma = unpackparam(theta)
        Z = length(alpha)
        S = size(sigma, 2)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        @assert Z == size(sigma, 1)
        @assert S == length(state_cutoff) + 1
        players = groupby(data, :id)

        exp_loglike = 0
        for (i, player) in enumerate(players)
            payoffs = 0
            Ti = nrow(player)
            for row in eachrow(player)
                period = row.period
                a = row.a
                delta_rd = row.delta_rd
                period = row.period
                if period != 1
                    state = get_state(row.prev_avg_a, state_cutoff)
                    prob = sigma[:,state]
                else
                    payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                    utility = alpha .+ delta_rd .* beta .+ lambda .* payoffs
                    prob = sigmoid.(utility)
                end
                for j in 1:Z
                    exp_loglike += r[i,j] * (logfinite(epsilon)(pj[j]) / Ti + (a >= cutoff) * logfinite(epsilon)(prob[j]) + (a < cutoff) * logfinite(epsilon)(1 - prob[j]))
                end
            end
        end
        return exp_loglike
    end
    
    function predict(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        Z = length(alpha)
        S = size(sigma, 2)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        @assert Z == size(sigma, 1)
        @assert S == length(state_cutoff) + 1
        
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = 0
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                period = row.period
                if period != 1
                    state = get_state(row.prev_avg_a, state_cutoff)
                    prob = sigma[j, state]
                else
                    payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                    utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                    prob = sigmoid(utility)
                end
                pred = ifelse(prob >= cutoff, 1, 0)
                push!(pred_df, (row.id, a, pred, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, dleta_rd, row.session, row.study))
            end
        end
        return pred_df
    end
    
    function accuracy(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        Z = length(alpha)
        S = size(sigma, 2)
        @assert Z == length(beta)
        @assert Z == length(lambda)
        @assert Z == size(sigma, 1)
        @assert S == length(state_cutoff) + 1
        acc = 0
        calc_responsibilities(theta)
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            payoffs = 0
            max_resp, j = findmax(r[i,:])
            for row in eachrow(player)
                a = row.a
                delta_rd = row.delta_rd
                period = row.period
                if period != 1
                    state = get_state(row.prev_avg_a, state_cutoff)
                    prob = sigma[j, state]
                else
                    payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                    utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                    prob = sigmoid(utility)
                end
                acc += (a >= cutoff) * (prob >= cutoff) + (a < cutoff) * (prob < cutoff)
            end
        end
        return acc / nrow(data)
    end

    function expected_loglike_fixed(free_params, fixed_params)
        alpha = free_params.alpha
        beta = free_params.beta
        lambda = free_params.lambda
        sigma = fixed_params.sigma
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
        return expected_loglike(theta)
    end

    #TODO: Implement log like
    function loglike(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        Z = length(pj)
        @assert length(alpha) == Z
        @assert length(beta) == Z
        @assert length(lambda) == Z
        @assert size(sigma, 1) == Z
        loglike = 0
        players = groupby(data, :id)
        for (i, player) in enumerate(players)
            for j in 1:Z
                ll_j = 0
                payoffs = 0
                for row in eachrow(player)
                    a = row.a
                    delta_rd = row.delta_rd
                    period = row.period
                    if period != 1
                        state = get_state(row.prev_avg_a, state_cutoff)
                        prob = sigma[j,state]
                    else
                        payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                        utility = alpha[j] + delta_rd * beta[j] + lambda[j] * payoffs
                        prob = sigmoid(utility)
                    end
                    ll_j += (a >= cutoff) * logfinite(epsilon)(prob) + (a < cutoff) * logfinite(epsilon)(1 - prob)
                end
                ll_j += logfinite(epsilon)(pj[j])
                loglike += logfinite(epsilon)(exp(ll_j))
            end
        end
        return loglike / nrow(data)
    end

    return (exp_ll = expected_loglike, unpack = unpackparam, pack = packparam, exp_ll_fixed = expected_loglike_fixed, calc_r = calc_responsibilities, calc_pj = calc_pj, accuracy = accuracy, predict = predict, loglike = loglike)
end

function mixture_irlsg_mult(data, pj::Vector{Float64}, r, cutoffs::Vector{Float64}, state_cutoff::Vector{Float64}, epsilon::Float64)
    function packparam(alpha::Matrix{Float64}, beta::Matrix{Float64}, lambda::Vector{Float64})
        S = length(state_cutoff) + 1
        Z = size(alpha, 1)
        K = size(alpha, 2) + 1
        @assert Z == size(beta, 1)
        @assert Z == length(lambda)
        @assert K == size(beta, 2) + 1
        @assert K == length(cutoffs) + 1
        sigma = zeros(Z, S, K - 1)
        data_non_init = filter(row ->row.period != 1, data)
        players = groupby(data_non_init, :id)
        I = length(players)
        n = zeros(I, S, K)
        calc_responsibilities(ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)) 
        # count for n matrix
        for (i, player) in enumerate(players)
            for row in eachrow(player)
                state = get_state(row.prev_avg_a, state_cutoff)
                ak = get_action(row.a, cutoffs)
                n[i, state, ak] += 1
            end
        end
        # calculate sigmas
        for j in 1:Z
            for s in 1:S
                for k in 1:K-1
                    num = 0
                    den = 0
                    for i in 1:I
                        num += r[i, j] * (n[i, s, k+1])
                        den += r[i, j] * (sum(n[i, s, :]))
                    end
                    sigma[j, s, k] = (num + 1) / (den + K)
                end
            end
        end

        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda, theta.sigma)
    end

    function calc_responsibilities(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
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
            #payoffs = zeros(K)
            for j in 1:Z
                payoffs = zeros(K)
                for row in eachrow(player)
                    period = row.period
                    a = row.a
                    delta_rd = row.delta_rd
                    if period != 1
                        state = get_state(row.prev_avg_a, state_cutoff)
                        probs = sigma[j,state,:]
                        insert!(probs, 1, 1 - sum(probs))
                    else
                        init_k = get_action(row.prev_super_game_init, cutoffs)
                        payoffs[init_k] += row.prev_super_game_payoff
                        utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                        probs = softmax_2(-utilities)
                        push!(probs, 1 - sum(probs))
                    end
                    ak = get_action(a, cutoffs)
                    log_lls[i, j] += logfinite(epsilon)(probs[ak])
                end
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
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
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
                    period = row.period
                    if period != 1
                        state = get_state(row.prev_avg_a, state_cutoff)
                        probs = sigma[j,state,:]
                        insert!(probs, 1, 1 - sum(probs))     
                    else
                        init_k = get_action(row.prev_super_game_init, cutoffs)
                        payoffs[init_k] += row.prev_super_game_payoff
                        utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                        probs = softmax_2(-utilities)
                        push!(probs, 1 - sum(probs))
                    end
                    ak = get_action(a, cutoffs)
                    ll_j += logfinite(epsilon)(probs[ak])
                end
                exp_loglike += r[i,j] * (logfinite(epsilon)(pj[j]) + ll_j)
            end
        end
        return exp_loglike
    end

    function predict(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        S = length(state_cutoff) + 1
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
                period = row.period
                if period != 1
                    state = get_state(row.prev_avg_a, state_cutoff)
                    probs = sigma[j,state,:]
                    insert!(probs, 1, 1 - sum(probs))
                else
                    init_k = get_action(row.prev_super_game_init, cutoffs)
                    payoffs[init_k] += row.prev_super_game_payoff
                    utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                    probs = softmax_2(-utilities)
                    push!(probs, 1 - sum(probs))
                end
                max_prob, pred = findmax(probs)
                push!(pred_df, (row.id, a, pred-1, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
            end
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
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
                period = row.period
                if period != 1
                    state = get_state(row.prev_avg_a, state_cutoff)
                    probs = sigma[j,state,:]
                    insert!(probs, 1, 1 - sum(probs))
                else
                    init_k = get_action(row.prev_super_game_init, cutoffs)
                    payoffs[init_k] += row.prev_super_game_payoff
                    utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                    probs = softmax_2(-utilities)
                    push!(probs, 1 - sum(probs))
                end
                max_prob, pred = findmax(probs)
                ak = get_action(a, cutoffs)
                acc += (ak == pred)
            end
        end
        return acc / nrow(data)
    end

    function expected_loglike_fixed(free_params, fixed_params)
        alpha = free_params.alpha
        beta = free_params.beta
        lambda = free_params.lambda
        sigma = fixed_params.sigma
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
        return expected_loglike(theta)
    end

    function loglike(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
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
                    period = row.period
                    if period != 1
                        state = get_state(row.prev_avg_a, state_cutoff)
                        probs = sigma[j,state,:]
                        insert!(probs, 1, 1 - sum(probs))
                    else
                        init_k = get_action(row.prev_super_game_init, cutoffs)
                        payoffs[init_k] += row.prev_super_game_payoff
                        utilities = alpha[j,:] .+ delta_rd * beta[j,:] .+ lambda[j] * (payoffs[K] .- payoffs[1:K-1])
                        probs = softmax_2(-utilities)
                        push!(probs, 1 - sum(probs))
                    end
                    ak = get_action(a, cutoffs)
                    ll_j += logfinite(epsilon)(probs[ak])
                end
                ll_j += logfinite(epsilon)(pj[j])
                loglike += logfinite(epsilon)(exp(ll_j))
            end
        end
        return loglike / nrow(data)
    end

    return (exp_ll = expected_loglike, unpack = unpackparam, pack = packparam, exp_ll_fixed = expected_loglike_fixed, calc_r = calc_responsibilities, calc_pj = calc_pj, accuracy = accuracy, predict = predict, loglike = loglike)
end
