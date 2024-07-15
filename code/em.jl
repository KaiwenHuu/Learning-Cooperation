using CSV
using ComponentArrays
using DataFrames
using FilePathsBase
using ForwardDiff
using Optim
using Test

include("utils.jl")
include("mixture_irlsg.jl")
include("mixture_reinforcementlearning.jl")

function em_rl(model, data, condition, action_ranges, alpha, beta, lambda, pj, r, eps, opt_alg, modelname, k, max_iter=100, tol=1e-6)
    data_train_copy = filter(condition, data)
    data_test_copy = filter(x -> !condition(x), data)

    exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = model(data_train_copy, pj, r, action_ranges, eps)
    theta_hat = pack(alpha, beta, lambda)
    pj_hat = pj
    combined_theta_hat = ComponentArray(pj = pj_hat, theta = theta_hat)
    calc_r(theta_hat)
    for iter in 1:max_iter
        @show prev_exp_ll = exp_ll(theta_hat)
        @show res = optimize(x->-exp_ll(x), theta_hat, opt_alg, autodiff=:forward)
        theta_new = res.minimizer
        pj_hat = calc_pj()
        @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
        @assert res.g_converged
        theta_hat = theta_new
        calc_r(theta_hat)
        @show curr_exp_ll = exp_ll(theta_hat)
        try
            @assert curr_exp_ll >= prev_exp_ll
        catch
            @warn "expected log likelihood went down"
            break
        end
        if abs(curr_exp_ll - prev_exp_ll) < tol
            break
        end
    end
    
    save_est_var(modelname, k, loglike, nrow(data_train_copy), theta_hat)

    exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = model(data, pj_hat, r, action_ranges, eps)
    calc_r(theta_hat)
    predictions = predict(theta_hat)

    exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = model(data_test_copy, pj_hat, r, action_ranges, eps)
    calc_r(theta_hat)
    test_likelihood = loglike(theta_hat)
    test_accuracy = accuracy(theta_hat)
    
    try
        @testset begin
            pred_test = filter(x ->!condition(x), predictions)
            @test pred_is_correct(pred_test, test_accuracy, action_ranges)
        end
    catch
        @warn "accuracy is wrong"
    end
    prediction_path = joinpath("..", "result", modelname, k)
    mkpath(prediction_path)
    prediction_csv_path = joinpath(prediction_path, "pred.csv")
    CSV.write(prediction_csv_path, predictions)

    return test_likelihood, test_accuracy
end

function em_irlsg(model, data, condition, action_ranges, state_cutoffs, alpha, beta, lambda, pj, r, eps, opt_alg, modelnames, k, max_iter=100, tol=1e-6)
    data_train_copy = filter(condition, data)
    data_test_copy = filter(x -> !condition(x), data)
    
    @assert length(state_cutoffs) == length(modelnames)
    S = length(state_cutoffs)
    test_likelihoods = zeros(S)
    test_accs = zeros(S)

    for s in 1:S
        loglike = nothing
        theta_hat = nothing
        #if s == 1
        exp_ll, unpack, pack, exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = model(data_train_copy, pj, r, action_ranges, state_cutoffs[s], eps)
        theta_hat = pack(alpha, beta, lambda)
        pj_hat = pj
        combined_theta_hat = ComponentArray(pj = pj_hat, theta = theta_hat)
        calc_r(theta_hat)
        for iter in 1:max_iter
            @show prev_exp_ll = exp_ll(theta_hat)
            alpha_hat, beta_hat, lambda_hat, sigma_hat = unpack(theta_hat)
            free_theta = ComponentArray(alpha = alpha_hat, beta = beta_hat, lambda = lambda_hat)
            fixed_theta = ComponentArray(sigma = sigma_hat)
            @show res = optimize(x->-exp_ll_fixed(x, fixed_theta), free_theta, opt_alg, autodiff=:forward)
            #TODO: confirm that this is converting alpha_hat and beta_hat to the correct data types: Vector or Matrix
            alpha_hat = convert(typeof(alpha), res.minimizer.alpha)
            beta_hat = convert(typeof(beta), res.minimizer.beta)
            lambda_hat = convert(Vector, res.minimizer.lambda)
            theta_new = pack(alpha_hat, beta_hat, lambda_hat)
            pj_hat = calc_pj()
            @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
            @assert res.g_converged
            theta_hat = theta_new
            calc_r(theta_hat)
            @show curr_exp_ll = exp_ll(theta_hat)
            try
                @assert curr_exp_ll >= prev_exp_ll
            catch
                @warn "expected log likelihood went down"
                break
            end
            if abs(curr_exp_ll - prev_exp_ll) < tol
                break
            end
        end
        
        save_est_var(modelnames[s], k, loglike, nrow(data_train_copy), theta_hat)

        exp_ll, unpack, pack, exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = model(data, pj_hat, r, action_ranges, state_cutoffs[s], eps)
        calc_r(theta_hat)
        predictions = predict(theta_hat)

        exp_ll, unpack, pack, exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = model(data_test_copy, pj_hat, r, action_ranges, state_cutoffs[s], eps)
        calc_r(theta_hat)
        test_likelihoods[s] = loglike(theta_hat)
        test_accs[s] = accuracy(theta_hat)
    
        try
            @testset begin
                pred_test = filter(x ->!condition(x), predictions)
                @test pred_is_correct(pred_test, test_accs[s], action_ranges)
            end
        catch
            @warn "accuracy is wrong"
        end
        prediction_path = joinpath("..", "result", modelnames[s], k)
        mkpath(prediction_path)
        prediction_csv_path = joinpath(prediction_path, "pred.csv")
        CSV.write(prediction_csv_path, predictions)
        #end
    end
    return test_likelihoods, test_accs
end
