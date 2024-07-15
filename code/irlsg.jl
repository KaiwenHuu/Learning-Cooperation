using DataFrames

include("utils.jl")

function irlsg(data, cutoff::Float64, state_cutoff::Vector{Float64}, epsilon::Float64)
    function packparam(alpha, beta, lambda)
        S = length(state_cutoff) + 1
        sigma = zeros(S)
        n = zeros(S)
        
        n[1] = count(row -> row.period != 1 && row.prev_avg_a < state_cutoff[1], eachrow(data))
        sigma[1] = count(row -> row.period != 1 && row.prev_avg_a < state_cutoff[1] && row.a >= cutoff, eachrow(data))

        n[S] = count(row -> row.period != 1 && state_cutoff[S-1] <= row.prev_avg_a, eachrow(data))
        sigma[S] = count(row -> row.period != 1 && state_cutoff[S-1] <= row.prev_avg_a && row.a >= cutoff, eachrow(data))

        for i in 2:S-1
            n[i] = count(row -> row.period != 1 && state_cutoff[i-1] <= row.prev_avg_a < state_cutoff[i], eachrow(data))
            sigma[i] = count(row -> row.period != 1 && state_cutoff[i-1] <= row.prev_avg_a < state_cutoff[i] && row.a >= cutoff, eachrow(data))
        end
        #TODO: Empirical Bayes
        sigma = (sigma .+ 1) ./ (n .+ 2)
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda, theta.sigma)
    end

    function loglike(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(sigma)
        curr_id = nothing
        loglikelihood = 0
        payoffs = 0
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = 0
            end
            a = row.a
            delta_rd = row.delta_rd
            period = row.period
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                prob = sigma[state]
            else
                payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                utility = alpha + delta_rd * beta + lambda * payoffs
                prob = sigmoid(utility)
            end
            loglikelihood += (a >= cutoff) * logfinite(epsilon)(prob) + (a < cutoff) * logfinite(epsilon)(1 - prob)
        end
        return loglikelihood/nrow(data)
    end

    function loglike_fixed(free_params, fixed_params)
        alpha = free_params.alpha
        beta = free_params.beta
        lambda = free_params.lambda
        sigma = fixed_params.sigma
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
        return loglike(theta)
    end

    function predict(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(sigma)
        curr_id = nothing
        payoffs = 0
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = 0
            end
            a = row.a
            delta_rd = row.delta_rd
            period = row.period
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                prob = sigma[state]
            else
                payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                utility = alpha + delta_rd * beta + lambda * payoffs
                prob = sigmoid(utility)
            end
            pred = ifelse(prob >= cutoff, 1, 0)
            push!(pred_df, (row.id, a, pred, period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(sigma)
        curr_id = nothing
        acc = 0
        payoffs = 0
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = 0
            end
            a = row.a
            delta_rd = row.delta_rd
            period = row.period
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                prob = sigma[state]
            else
                payoffs += ifelse(row.prev_super_game_init >= cutoff, row.prev_super_game_payoff, -row.prev_super_game_payoff)
                utility = alpha + delta_rd * beta + lambda * payoffs
                prob = sigmoid(utility)
            end
            acc += (a >= cutoff) * (prob >= cutoff) + (a < cutoff) * (prob < cutoff)
        end
        return acc / nrow(data)
    end
    
    return(loglike = loglike, unpack = unpackparam, pack = packparam, loglike_fixed = loglike_fixed, accuracy = accuracy, predict = predict)
end

function irlsg_mult(data, cutoffs::Vector{Float64}, state_cutoff::Vector{Float64}, epsilon::Float64)
    function packparam(alpha, beta, lambda)
        S = length(state_cutoff) + 1
        Z = length(cutoffs)
        sigma = zeros(S, Z)
        n = zeros(S)
        
        n[1] = count(row -> row.period != 1 && row.prev_avg_a < state_cutoff[1], eachrow(data))
        for j in Z:-1:1
            if j == Z
                sigma[1, j] = count(row -> row.period != 1 && row.prev_avg_a < state_cutoff[1] && cutoffs[j] <= row.a, eachrow(data))
            else
                sigma[1, j] = count(row -> row.period != 1 && row.prev_avg_a < state_cutoff[1] && cutoffs[j] <= row.a && row.a < cutoffs[j+1], eachrow(data))
            end
        end

        n[S] = count(row -> row.period != 1 && state_cutoff[S-1] <= row.prev_avg_a, eachrow(data))
        for j in Z:-1:1
            if j == Z
                sigma[S, j] = count(row -> row.period != 1 && state_cutoff[S-1] <= row.prev_avg_a && cutoffs[j] <= row.a, eachrow(data))
            else
                sigma[S, j] = count(row -> row.period != 1 && state_cutoff[S-1] <= row.prev_avg_a && cutoffs[j] <= row.a && row.a < cutoffs[j+1], eachrow(data))
            end
        end

        for i in 2:S-1
            n[i] = count(row -> state_cutoff[i-1] <= row.prev_avg_a && row.prev_avg_a < state_cutoff[i], eachrow(data))
            for j in Z:-1:1
                if j == Z
                    sigma[i, j] = count(row -> row.period != 1 && state_cutoff[i-1] <= row.prev_avg_a && row.prev_avg_a < state_cutoff[i] && cutoffs[j] <= row.a, eachrow(data))
                else
                    sigma[i, j] = count(row -> row.period != 1 && state_cutoff[i-1] <= row.prev_avg_a && row.prev_avg_a < state_cutoff[i] && cutoffs[j] <= row.a && row.a < cutoffs[j+1], eachrow(data))
                end
            end
        end

        for i in 1:S
            sigma[i, :] .= (sigma[i, :] .+ 1) ./ (n[i] .+ (Z + 1))
        end

        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda, theta.sigma)
    end

    function loglike(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
        curr_id = nothing
        loglikelihood = 0
        K = length(alpha) + 1
        payoffs = zeros(K)
        @assert K == length(cutoffs) + 1
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = zeros(K)
            end
            a = row.a
            delta_rd = row.delta_rd
            period = row.period
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                probs = sigma[state,:]
                insert!(probs, 1, 1 - sum(probs))
            else
                init_k = get_action(row.prev_super_game_init, cutoffs)
                payoffs[init_k] += row.prev_super_game_payoff
                utilities = alpha .+ delta_rd .* beta .+ lambda * (payoffs[K] .- payoffs[1:K-1]) 
                probs = softmax_2(-utilities)
                push!(probs, 1 - sum(probs))
            end
            ak = get_action(a, cutoffs)
            loglikelihood += logfinite(epsilon)(probs[ak])
        end
        return loglikelihood / nrow(data)
    end

    function loglike_fixed(free_params, fixed_params)
        alpha = free_params.alpha
        beta = free_params.beta
        lambda = free_params.lambda
        sigma = fixed_params.sigma
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda, sigma = sigma)
        return loglike(theta)
    end

    function predict(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
        curr_id = nothing
        K = length(alpha) + 1
        @assert K == length(cutoffs) + 1
        payoffs = zeros(K)
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = zeros(K)
            end
            a = row.a
            delta_rd = row.delta_rd
            period = row.period
            probs = nothing
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                probs = sigma[state,:]
                insert!(probs, 1, 1 - sum(probs))
            else
                init_k = get_action(row.prev_super_game_init, cutoffs)
                payoffs[init_k] += row.prev_super_game_payoff
                utilities = alpha .+ delta_rd .* beta .+ lambda * (payoffs[K] .- payoffs[1:K-1])
                probs = softmax_2(-utilities)
                push!(probs, 1 - sum(probs))
            end
            max_prob, pred = findmax(probs)
            push!(pred_df, (row.id, a, pred - 1, period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda, sigma = unpackparam(theta)
        S = length(state_cutoff) + 1
        curr_id = nothing
        acc = 0
        K = length(alpha) + 1
        payoffs = zeros(K)
        @assert K == length(cutoffs) + 1
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = zeros(K)
            end
            a = row.a
            delta_rd = row.delta_rd 
            period = row.period
            probs = nothing
            if period != 1
                state = get_state(row.prev_avg_a, state_cutoff)
                probs = sigma[state,:]
                insert!(probs, 1, 1 - sum(probs))
            else
                init_k = get_action(row.prev_super_game_init, cutoffs)
                payoffs[init_k] += row.prev_super_game_payoff
                utilities = alpha .+ delta_rd .* beta .+ lambda * (payoffs[K] .- payoffs[1:K-1])
                probs = softmax_2(-utilities)
                push!(probs, 1 - sum(probs))
            end
            max_prob, pred = findmax(probs)
            ak = get_action(a, cutoffs)
            acc += (ak == pred)
        end
        return acc / nrow(data)
    end

    return(loglike = loglike, unpack = unpackparam, pack = packparam, loglike_fixed = loglike_fixed, accuracy = accuracy, predict = predict)
end

