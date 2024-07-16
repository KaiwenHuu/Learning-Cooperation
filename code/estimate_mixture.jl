using ComponentArrays
using DataFrames
using FilePathsBase
using ForwardDiff
using Optim
using Test
using Random

include("em.jl")
include("utils.jl")
include("mixture_reinforcementlearning.jl")
include("mixture_irlsg.jl")
csv_file_path = joinpath("..", "data", "cleaned_data.csv")
df_all = CSV.read(csv_file_path, DataFrame)
df = filter(row -> row.is_probabilistic == 1, df_all)
df_lugovskyy = filter(row -> row.study == "lugovskyy et al (2017)", df)

# Simulated Data with Two classes
# PI = 1/5: P(Z=1) = 1/5 and P(Z=2) = 4/5
# alpha1 = -0.2, beta1 = 0.5, lam1 = 0.2
# alpha2 = -0.4, beta2 = 0.1, lam2 = 0.05
#sim_path = joinpath("..", "data", "test_mixture_sim.csv")

Random.seed!(1234)

cond = row -> row.study == "lugovskyy et al (2017)"
lam = [rand(), rand()]
alpha = [rand(); rand();;]
beta = [rand(); rand();;]
#@show alpha = fill(rand(), (2, 1))
#@show beta = fill(rand(), (2, 1))
alpha_3 = fill(rand(), (2, 2))
beta_3 = fill(rand(), (2, 2))
#sigma = fill(1/2, (2,4,1))
#sigma_3 = fill(1/2, (2,4,2))
#@show sigma
#@show sigma_3
e = 1e-6
opt_alg = Adam(alpha=0.01)
cutoff = 0.5

pj = [1/2, 1/2]
r = nothing

likelihoods = DataFrame(z = Int[], k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])
accuracies = DataFrame(z = Int[], k = Int[], irlsg_4 = Float64[], irlsg_5 = Float64[], irlsg_10 = Float64[], reinforcement_learning = Float64[])
for K in 2:11
    cond = row -> row.study == "lugovskyy et al (2017)"
    e = 1e-6
    opt_alg = Adam(alpha=0.01)
    cutoffs = zeros(K-1)
    four_state = [1/4, 2/4, 3/4]
    five_state = [1/5, 2/5, 3/5, 4/5]
    ten_state = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10]
    state_bins = [four_state, five_state, ten_state]
    for k in 1:K-1
        cutoffs[k] = k/K
    end
    for Z in 2:2
        pathnames = ["cross_treat/mixture_$Z/irlsg_state_4", "cross_treat/mixture_$Z/irlsg_state_5", "cross_treat/mixture_$Z/irlsg_state_10"]
        #TODO: For some reason there can be bad initial parameters.
        lam = rand(Z)
        alpha = rand(Z, K-1)
        beta = rand(Z, K-1)
        pj = fill(1/Z, Z)
        println("lam: $lam")
        println("alpha: $alpha")
        println("beta: $beta")
        println("pj: $pj")

        rl_ll, rl_acc = em_rl(mixture_rl_mult, df, cond, cutoffs, alpha, beta, lam, pj, nothing, e, opt_alg, "cross_treat/mixture_$Z/reinforcement_learning", "K_$K")
        lls, accs = em_irlsg(mixture_irlsg_mult, df, cond, cutoffs, state_bins, alpha, beta, lam, pj, nothing, e, opt_alg, pathnames, "K_$K")
        @show lls, accs
        @show rl_ll, rl_acc
        
        push!(likelihoods, (Z, K, lls[1], lls[2], lls[3], rl_ll))
        push!(accuracies, (Z, K, accs[1], accs[2], accs[3], rl_acc))
        likelihood_output_path = joinpath("..", "result", "cross_treat/likelihoods")
        mkpath(likelihood_output_path)
        accuracy_output_path = joinpath("..", "result", "cross_treat/accuracy")
        mkpath(accuracy_output_path)
        likelihood_csv_path = joinpath(likelihood_output_path, "mixture_likelihood.csv")
        CSV.write(likelihood_csv_path, likelihoods)
        println("all mixture likelihoods saved to $likelihood_csv_path")
        accuracy_csv_path = joinpath(accuracy_output_path, "mixture_accuracy.csv")
        CSV.write(accuracy_csv_path, accuracies)
        println("all accuracies saved to $accuracy_csv_path")
    end
end

function em_rl_test(data, init_alpha, init_beta, init_lam, init_pj, init_r, cutoff, epsilon, max_iter=10, tol=1e-6)
    exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = mixture_rl(data, init_pj, init_r, cutoff, epsilon)
    theta = pack(init_alpha, init_beta, init_lam)
    pj_hat = init_pj
    calc_r(theta)
    for iter in 1:max_iter
        @show prev_exp_ll = exp_ll(theta)
        res = optimize(x->-exp_ll(x), theta, opt_alg, autodiff=:forward)
        theta_new = res.minimizer
        pj_hat = calc_pj()
        @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
        @assert res.g_converged
        theta = theta_new
        calc_r(theta)
        @show curr_exp_ll = exp_ll(theta)
        @assert curr_exp_ll >= prev_exp_ll
        if abs(curr_exp_ll - prev_exp_ll) < tol
            break
        end
    end
    return theta, pj_hat
end

function em_rl_mult_test(data, init_alpha, init_beta, init_lam, init_pj, init_r, cutoff, epsilon, max_iter=100, tol=1e-6)
    exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = mixture_rl_mult(data, init_pj, init_r, cutoff, epsilon)
    theta = pack(init_alpha, init_beta, init_lam)
    pj_hat = init_pj
    calc_r(theta)
    for iter in 1:max_iter
        @show prev_exp_ll = exp_ll(theta)
        res = optimize(x->-exp_ll(x), theta, opt_alg, autodiff=:forward)
        theta_new = res.minimizer
        pj_hat = calc_pj()
        @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
        @assert res.g_converged
        theta = theta_new
        calc_r(theta)
        @show curr_exp_ll = exp_ll(theta)
        @assert curr_exp_ll >= prev_exp_ll
        if abs(curr_exp_ll - prev_exp_ll) < tol
            break
        end
    end
    return theta, pj_hat
end

function em_irlsg_test(data, init_alpha, init_beta, init_lam, init_pj, init_r, cutoff, state_cutoff, epsilon, max_iter=100, tol=1e-6)
    exp_ll, unpack, pack, exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = mixture_irlsg(data, init_pj, init_r, cutoff, state_cutoff, epsilon)
    theta = pack(init_alpha, init_beta, init_lam)
    @show theta
    pj_hat = init_pj
    calc_r(theta)
    @show exp_ll(theta)
    for iter in 1:max_iter
        @show prev_exp_ll = exp_ll(theta)
        alpha, beta, lambda, sigma = unpack(theta)
        free_theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
        fixed_theta = ComponentArray(sigma = sigma)
        res = optimize(x->-exp_ll_fixed(x, fixed_theta), free_theta, opt_alg, autodiff=:forward)
        alpha_hat = convert(Vector, res.minimizer.alpha)
        beta_hat = convert(Vector, res.minimizer.beta)
        lambda_hat = convert(Vector, res.minimizer.lambda)
        theta_new = pack(alpha_hat, beta_hat, lambda_hat)
        pj = calc_pj()
        @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
        @assert res.g_converged
        theta = theta_new
        calc_r(theta)
        @show curr_exp_ll = exp_ll(theta)
        @assert curr_exp_ll >= prev_exp_ll
        if abs(curr_exp_ll - prev_exp_ll) < tol
            break
        end
    end
    return theta, pj
end

function em_irlsg_mult_test(data, init_alpha, init_beta, init_lam, init_pj, init_r, cutoff, state_cutoff, epsilon, max_iter=100, tol=1e-6)
    exp_ll, unpack, pack, exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = mixture_irlsg_mult(data, init_pj, init_r, cutoff, state_cutoff, epsilon)
    theta = pack(init_alpha, init_beta, init_lam)
    @show theta
    pj_hat = init_pj
    calc_r(theta)
    @show exp_ll(theta)
    for iter in 1:max_iter
        @show prev_exp_ll = exp_ll(theta)
        alpha, beta, lambda, sigma = unpack(theta)
        free_theta = ComponentArray(alpha = alpha, beta = beta, lambda = lambda)
        fixed_theta = ComponentArray(sigma = sigma)
        res = optimize(x->-exp_ll_fixed(x, fixed_theta), free_theta, opt_alg, autodiff=:forward)
        alpha_hat = convert(Matrix, res.minimizer.alpha)
        beta_hat = convert(Matrix, res.minimizer.beta)
        lambda_hat = convert(Vector, res.minimizer.lambda)
        theta_new = pack(alpha_hat, beta_hat, lambda_hat)
        pj = calc_pj()
        @assert all(isfinite.(ForwardDiff.gradient(loglike, theta_new)))
        @assert res.g_converged
        pj = calc_pj()
        theta = theta_new
        calc_r(theta)
        @show curr_exp_ll = exp_ll(theta)
        @assert curr_exp_ll >= prev_exp_ll
        if abs(curr_exp_ll - prev_exp_ll) < tol
            break
        end
    end
    return theta, pj
end

df_train = filter(row -> row.study == "lugovskyy et al (2017)", df)
df_test = filter(row -> row.study != "lugovskyy et al (2017)", df)

theta_hat, pj_hat = em_rl_test(df_train, alpha, beta, lam, pj, r, 0.5, e)
exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = mixture_rl(df_test, pj_hat, nothing, 0.5, e)
acc = accuracy(theta_hat)
println("Binary: Accuracy of k = 2 for RL with Z = 2 $acc")

theta_hat, pj_hat = em_rl_mult_test(df_train, alpha, beta, lam, pj, r, [0.5], e)
exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = mixture_rl_mult(df_test, pj_hat, nothing, [0.5], e)
acc = accuracy(theta_hat)
println("Mult: Accuracy of k = 2 for RL with Z = 2 $acc")

theta_hat, pj_hat = em_rl_mult_test(df_train, alpha_3, beta_3, lam, pj, r, [1/3, 2/3], e)
exp_ll, pack, unpack, calc_r, calc_pj, accuracy, predict, loglike = mixture_rl_mult(df_test, pj_hat, nothing, [1/3, 2/3], e)
acc = accuracy(theta_hat)
println("Mult: Accuracy of k = 3 for RL with Z = 2 $acc")

theta_hat, pj_hat = em_irlsg_test(df_train, vec(alpha), vec(beta), lam, pj, r, 0.5, [1/4, 2/4, 3/4], e)
exp_ll, pack, unpack,exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = mixture_irlsg(df_test, pj_hat, nothing, 0.5, [1/4, 2/4, 3/4], e)
acc = accuracy(theta_hat)
println("Binary: Accuracy of k = 2 for IRLSG with 4 states and Z = 2 $acc")
@show loglike(theta_hat)

theta_hat, pj_hat = em_irlsg_mult_test(df_train, alpha, beta, lam, pj, r, [0.5], [1/4, 2/4, 3/4], e)
exp_ll, pack, unpack,exp_ll_fixed, calc_r, calc_pj, accuracy, predict, loglike = mixture_irlsg_mult(df_test, pj_hat, nothing, [0.5], [1/4, 2/4, 3/4], e)
calc_r(theta)
acc = accuracy(theta_hat)
println("Mult: Accuracy of k = 2 for IRLSG with 4 states and Z = 2 $acc")
@show loglike(theta_hat, pj_hat)
