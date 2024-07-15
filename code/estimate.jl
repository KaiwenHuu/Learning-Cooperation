using CSV
using ComponentArrays
using DataFrames
using FilePathsBase
using ForwardDiff
using Optim
using Test

include("utils.jl")
include("irlsg.jl")
include("reinforcementlearning.jl")

csv_file_path = joinpath("..", "data", "cleaned_data.csv")
df_all = CSV.read(csv_file_path, DataFrame)
df = filter(row -> row.is_probabilistic == 1, df_all)
df_lugovskyy = filter(row -> row.study == "lugovskyy et al (2017)", df)

function run_optimization_irlsg(model, data, condition, action_ranges, state_cutoffs, alpha, beta, lambda, eps, opt_alg, modelnames, k, evaluate=true)
    data_train_copy = filter(condition, data)
    data_test_copy = filter(x -> !condition(x), data)
    n = length(alpha)

    @assert length(state_cutoffs) == length(modelnames)
    S = length(state_cutoffs)
    test_likelihoods = zeros(S)
    test_accs = zeros(S)

    alpha_hat = nothing
    beta_hat = nothing
    lambda_hat = nothing
    for s in 1:S
        loglike = nothing
        theta_hat = nothing
        if s == 1
            loglike, unpack, pack, loglike_fixed, accuracy, predict = model(data_train_copy, action_ranges, state_cutoffs[s], eps)
            theta = pack(alpha, beta, lambda)
            alpha, beta, lambda, sigma = unpack(theta)
            free_theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
            fixed_theta = ComponentArray(sigma = sigma)
            @show res = optimize(x->-loglike_fixed(x, fixed_theta), free_theta, opt_alg, autodiff=:forward)
            alpha_hat = res.minimizer.alpha
            beta_hat = res.minimizer.beta
            lambda_hat = res.minimizer.lambda
            theta_hat = pack(alpha_hat, beta_hat, lambda_hat)
            @testset begin
                @test all(isfinite.(ForwardDiff.gradient(loglike, theta_hat)))
                @test res.g_converged
            end
        else
            loglike, unpack, pack, loglike_fixed, accuracy, predict = model(data_train_copy, action_ranges, state_cutoffs[s], eps)
            theta_hat = pack(alpha_hat, beta_hat, lambda_hat)
        end
        save_est_var(modelnames[s], k, loglike, nrow(data_train_copy), theta_hat)

        if evaluate
            loglike, unpack, pack, loglike_fixed, accuracy, predict = model(data, action_ranges, state_cutoffs[s], eps)
        
            predictions = predict(theta_hat)

            loglike, unpack, pack, loglike_fixed, accuracy, predict = model(data_test_copy, action_ranges, state_cutoffs[s], eps)
            test_likelihoods[s] = -loglike(theta_hat)
            test_accs[s] = accuracy(theta_hat)
            try
                @testset begin
                    pred_test = filter(x -> !condition(x), predictions)
                    @test pred_is_correct(pred_test, test_accs[s], action_ranges)
                end
            catch
                @warn "accuracy is wrong"
            end
            prediction_path = joinpath("..", "result", modelnames[s], k)
            mkpath(prediction_path)
            prediction_csv_path = joinpath(prediction_path, "pred.csv")
            CSV.write(prediction_csv_path, predictions)
        end
    end
    return test_likelihoods, test_accs
end

function run_optimization_reinforcementlearning(model, data, condition, action_ranges, alpha, beta, lambda, eps, opt_alg, modelname, k, evaluate=true)
    data_train_copy = filter(condition, data)
    data_test_copy = filter(x -> !condition(x), data)
    n = length(alpha)
    loglike, unpack, pack, accuracy, predict = model(data_train_copy, action_ranges, eps)
    theta = pack(alpha, beta, lambda)
    theta_hat = nothing
    @show res = optimize(x->-loglike(x), theta, opt_alg, autodiff=:forward)
    theta_hat = res.minimizer
    @testset begin
        @test all(isfinite.(ForwardDiff.gradient(loglike, theta_hat)))
        @test res.g_converged
    end
    save_est_var(modelname, k, loglike, nrow(data_train_copy), theta_hat)
    test_likelihood = 0
    test_accuracy = 0
    
    if evaluate
        loglike, unpack, pack, accuracy, predict = model(data, action_ranges, eps)
        predictions = predict(theta_hat)

        loglike, unpack, pack, accuracy, predict = model(data_test_copy, action_ranges, eps)
        test_likelihood = -loglike(theta_hat)
        test_accuracy = accuracy(theta_hat)
        try
            @testset begin
                pred_test = filter(x -> !condition(x), predictions)
                @test pred_is_correct(pred_test, test_accuracy, action_ranges)
            end
        catch
            @warn "accuracy be wrong"
        end
        prediction_path = joinpath("..", "result", modelname, k)
        mkpath(prediction_path)
        prediction_csv_path = joinpath(prediction_path, "pred.csv")
        CSV.write(prediction_csv_path, predictions)
    end
    return test_likelihood, test_accuracy
end


losses = DataFrame(k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])

accuracies = DataFrame(k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])

for K in 2:11
    # condition for all probabilistic treatments
    cond = row -> row.is_probabilistic == 1
    lam = 0.1
    e = 1e-6
    opt_alg = Adam(alpha=0.01)
    cutoffs = zeros(K-1)
    four_state = [1/4, 2/4, 3/4]
    five_state = [1/5, 2/5, 3/5, 4/5]
    ten_state = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10]
    state_bins = [four_state, five_state, ten_state]
    pathnames = ["all_prob/irlsg_state_4", "all_prob/irlsg_state_5", "all_prob/irlsg_state_10"]
    for k in 1:K-1
        cutoffs[k] = k/K
    end
    run_optimization_irlsg(irlsg_mult, df_all, cond, cutoffs, state_bins, ones(K-1), ones(K-1), lam, e, opt_alg, pathnames, "K_$K", false)
    run_optimization_reinforcementlearning(reinforcementlearning_mult, df_all, cond, cutoffs, ones(K-1), ones(K-1), lam, e, opt_alg, "all_prob/reinforcement_learning", "K_$K", false)
end

#for K in 2:26
#    # condition for all lugovskyy studies
#    cond = row -> row.study == "lugovskyy et al (2017)"
#    lam = 0.1
#    e = 1e-6
#    opt_alg = Adam(alpha=0.01)
#    cutoffs = zeros(K-1)
#    four_state = [1/4, 2/4, 3/4]
#    five_state = [1/5, 2/5, 3/5, 4/5]
#    ten_state = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10]
#    state_bins = [four_state, five_state, ten_state]
#    pathnames = ["mix/irlsg_state_4", "mix/irlsg_state_5", "mix/irlsg_state_10"]
#    for k in 1:K-1
#        cutoffs[k] = k/K
#    end
#    run_optimization_irlsg(irlsg_mult, df_all, cond, cutoffs, state_bins, ones(K-1), ones(K-1), lam, e, opt_alg, pathnames, "K_$K", false)
#    run_optimization_reinforcementlearning(reinforcementlearning_mult, df_all, cond, cutoffs, ones(K-1), ones(K-1), lam, e, opt_alg, "mix/reinforcement_learning", "K_$K", false)
#end

for K in 2:11
    cond = row -> row.study == "lugovskyy et al (2017)"
    lam = 0.1
    e = 1e-6
    opt_alg = Adam(alpha=0.01)
    cutoffs = zeros(K-1)
    four_state = [1/4, 2/4, 3/4]
    five_state = [1/5, 2/5, 3/5, 4/5]
    ten_state = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10]
    state_bins = [four_state, five_state, ten_state]
    pathnames = ["cross_treat/irlsg_state_4", "cross_treat/irlsg_state_5", "cross_treat/irlsg_state_10"]
    for k in 1:K-1
        cutoffs[k] = k/K
    end
    lls, accs = run_optimization_irlsg(irlsg_mult, df, cond, cutoffs, state_bins, ones(K-1), ones(K-1), lam, e, opt_alg, pathnames, "K_$K")
    rl_ll, rl_acc = run_optimization_reinforcementlearning(reinforcementlearning_mult, df, cond, cutoffs, ones(K-1), ones(K-1), lam, e, opt_alg, "cross_treat/reinforcement_learning", "K_$K")
    
    push!(losses, (K, lls[1], lls[2], lls[3], rl_ll))
    push!(accuracies, (K, accs[1], accs[2], accs[3], rl_acc))
    
    loss_output_path = joinpath("..", "result", "cross_treat/loss")
    mkpath(loss_output_path)
    accuracy_output_path = joinpath("..", "result", "cross_treat/accuracy")
    mkpath(accuracy_output_path)
    
    loss_csv_path = joinpath(loss_output_path, "loss.csv")
    CSV.write(loss_csv_path, losses)
    println("all losses saved to $loss_csv_path")
    accuracy_csv_path = joinpath(accuracy_output_path, "accuracy.csv")
    CSV.write(accuracy_csv_path, accuracies)
    println("all accuracies saved to $accuracy_csv_path")
end

losses = DataFrame(k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])
accuracies = DataFrame(k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])

for K in 2:26
    cond = row -> row.session != 3
    lam = 0.1
    e = 1e-6
    opt_alg = Adam(alpha=0.01)
    cutoffs = zeros(K-1)
    four_state = [1/4, 2/4, 3/4]
    five_state = [1/5, 2/5, 3/5, 4/5]
    ten_state = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10]
    state_bins = [four_state, five_state, ten_state]
    pathnames = ["in_treat/irlsg_state_4", "in_treat/irlsg_state_5", "in_treat/irlsg_state_10"]
    for k in 1:K-1
        cutoffs[k] = k/K
    end
    lls, accs = run_optimization_irlsg(irlsg_mult, df_lugovskyy, cond, cutoffs, state_bins, ones(K-1), ones(K-1), lam, e, opt_alg, pathnames, "K_$K")
    rl_ll, rl_acc  = run_optimization_reinforcementlearning(reinforcementlearning_mult, df_lugovskyy, cond, cutoffs, ones(K-1), ones(K-1), lam, e, opt_alg, "in_treat/reinforcement_learning", "K_$K")

    push!(losses, (K, lls[1], lls[2], lls[3], rl_ll))
    push!(accuracies, (K, accs[1], accs[2], accs[3], rl_acc))

    if K == 2
        lls_val, accs_val = run_optimization_irlsg(irlsg, df_lugovskyy, cond, 1/K, state_bins, 1, 1, lam, e, opt_alg, pathnames, "K_validate_$K")
        rl_ll_val, rl_acc_val = run_optimization_reinforcementlearning(reinforcementlearning, df_lugovskyy, cond, 1/K, 1, 1, lam, e, opt_alg, "in_treat/reinforcement_learning", "K_validate_$K")
        @testset begin
            @test all(lls_val .≈ lls)
            @test all(accs_val .≈ accs)
            @test rl_ll_val ≈ rl_ll
            @test rl_acc_val ≈ rl_acc
        end
    end

    loss_output_path = joinpath("..", "result", "in_treat/loss")
    mkpath(loss_output_path)
    accuracy_output_path = joinpath("..", "result", "in_treat/accuracy")
    mkpath(accuracy_output_path)

    loss_csv_path = joinpath(loss_output_path, "loss.csv")
    CSV.write(loss_csv_path, losses)
    println("all losses saved to $loss_csv_path")
    accuracy_csv_path = joinpath(accuracy_output_path, "accuracy.csv")
    CSV.write(accuracy_csv_path, accuracies)
    println("all accuracies saved to $accuracy_csv_path")
end

