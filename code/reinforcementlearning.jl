using DataFrames

include("utils.jl")

function reinforcementlearning(data, cutoff, epsilon)
    function packparam(alpha, beta, lambda)
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda)
    end

    function loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
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
            utility = alpha + delta_rd * beta + lambda * payoffs
            prob = sigmoid(utility)
            loglikelihood += (a >= cutoff) * logfinite(epsilon)(prob) + (a < cutoff) * logfinite(epsilon)(1 - prob)
            payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
        end
        return loglikelihood / nrow(data)
    end

    function predict(theta)
        alpha, beta, lambda = unpackparam(theta)
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
            utility = alpha + delta_rd * beta + lambda * payoffs
            prob = sigmoid(utility)
            pred = ifelse(prob >= cutoff, 1, 0)
            push!(pred_df, (row.id, a, pred, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
            payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda = unpackparam(theta)
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
            utility = alpha + delta_rd * beta + lambda * payoffs
            prob = sigmoid(utility)
            acc += (a >= cutoff) * (prob >= cutoff) + (a < cutoff) * (prob < cutoff)
            payoffs += ifelse(a >= cutoff, row.payoff, -row.payoff)
        end
        return acc / nrow(data)
    end

    return (loglike = loglike, unpack = unpackparam, pack = packparam, accuracy = accuracy, predict = predict)
end

function reinforcementlearning_mult(data, cutoffs, epsilon)
    function packparam(alpha, beta, lambda)
        theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
    end

    function unpackparam(theta)
        @views (theta.alpha, theta.beta, theta.lambda)
    end

    function loglike(theta)
        alpha, beta, lambda = unpackparam(theta)
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
            utilities = alpha .+ delta_rd * beta .+ lambda * (payoffs[K] .- payoffs[1:K-1])
            probs = softmax_2(-utilities)
            push!(probs, 1 - sum(probs))
            ak = get_action(a, cutoffs)
            loglikelihood += logfinite(epsilon)(probs[ak])
            payoffs[ak] += row.payoff
        end
        return loglikelihood / nrow(data)
    end

    function predict(theta)
        alpha, beta, lambda = unpackparam(theta)
        curr_id = nothing
        K = length(alpha) + 1
        payoffs = zeros(K)
        @assert K == length(cutoffs) + 1
        pred_df = DataFrame(id = Int[], a = Float64[], a_hat = Int[], period = Int[], sequence = Int[], round = Int[], mpcr = Float64[], c = Float64[], n = Int[], delta_rd = Float64[], session = Int[], study = String[])
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = zeros(K)
            end
            a = row.a
            delta_rd = row.delta_rd
            utilities = alpha .+ delta_rd * beta .+ lambda * (payoffs[K] .- payoffs[1:K-1])
            probs = softmax_2(-utilities)
            push!(probs, 1 - sum(probs))
            max_prob, pred = findmax(probs)
            push!(pred_df, (row.id, a, pred - 1, row.period, row.sequence, row.round, row.mpcr, row.c, row.n, delta_rd, row.session, row.study))
            ak = get_action(a, cutoffs)
            payoffs[ak] += row.payoff
        end
        return pred_df
    end

    function accuracy(theta)
        alpha, beta, lambda = unpackparam(theta)
        curr_id = nothing
        acc = 0
        K = length(alpha) + 1
        @assert K == length(cutoffs) + 1
        payoffs = zeros(K)
        for row in eachrow(data)
            if row.id != curr_id
                curr_id = row.id
                payoffs = zeros(K)
            end
            a = row.a
            delta_rd = row.delta_rd
            utilities = alpha .+ delta_rd * beta .+ lambda * (payoffs[K] .- payoffs[1:K-1])
            probs = softmax_2(-utilities)
            push!(probs, 1 - sum(probs))
            max_prob, pred = findmax(probs)
            ak = get_action(a, cutoffs)
            acc += (ak == pred)
            payoffs[ak] += row.payoff
        end
        return acc / nrow(data)
    end

    return (loglike = loglike, unpack = unpackparam, pack = packparam, accuracy = accuracy, predict = predict)
end

