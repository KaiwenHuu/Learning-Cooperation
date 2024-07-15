using CSV
using CategoricalArrays
using DataFrames
using Distributions
using FilePathsBase
using LinearAlgebra
using PrettyTables
using StatsBase
using Tables

function categorize(a, cutoffs)
    extended_cutoffs = [0; cutoffs; 1]
    categories = cut(a, extended_cutoffs, labels=0:length(cutoffs), extend = true)
    return categories
end

function count_values(z, K)
    counts = zeros(K)
    for i in 1:length(z)
        for j in 1:K
            if z[i] == j
                counts[j] += 1
            end
        end
    end
    return counts
    #counts = countmap(z)
    #return [get(counts, k, 0) for k in 1:K]
end

function get_state(a, state_cutoffs)
    S = length(state_cutoffs)
    conditions = [a < state_cutoffs[1]]
    append!(conditions, [state_cutoffs[s-1] <= a < state_cutoffs[s] for s in 2:S])
    append!(conditions, [state_cutoffs[S] <= a])
    return findfirst(conditions)
end

function get_action(a, action_cutoffs)
    K = length(action_cutoffs)
    conditions = [a < action_cutoffs[1]]
    append!(conditions, [action_cutoffs[k-1] <= a < action_cutoffs[k] for k in 2:K])
    append!(conditions, [action_cutoffs[K] <= a])
    return findfirst(conditions)
end

function p_values(est, se)
    z_scores = est ./ se
    p_values = 2 .* (1 .- cdf.(Normal(0, 1), abs.(z_scores)))
    return p_values
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function softmax_util(x)
    return exp.(x) ./ sum(exp.(x))
end

function softmax_2(x)
    return exp.(x) ./ (1 .+ sum(exp.(x)))
end

function logfinite(cut)
    lc = log(cut)
    dlc = 1.0/cut
    function(x)
        if (x>=cut)
            log(x)
        else
            lc + (x-cut)*dlc
        end
    end
end

function weighted_avg(x)
    if length(x) == 0
        return 0
    else
        weights = softmax_util(x)
        return dot(weights, x)
    end
end

function save_est_var(model, k, loglike, n, theta_hat)
    H = ForwardDiff.hessian(loglike, theta_hat)
    Var = try
        -inv(H)./n
    catch
        @warn "Hessian is singular, se are incorrect"
        -inv(H - I)./n
    end
    header = ["Estimate", "(SE)"]
    values = try
        hcat(theta_hat, sqrt.(diag(Var)))
    catch
        @warn "There are negative variances, se are incorrect"
        hcat(theta_hat, sqrt.(complex(diag(Var))))
    end
    fmter = (v,i,j) -> (j==2) ? "($(round(v,digits=4)))" : round(v, digits=3)
    hl = ()
    tbl = pretty_table(values,
                       header=header,
                       row_labels=ComponentArrays.labels(theta_hat),
                       highlighters=hl,
                       formatters=tuple(fmter))
    df_values = DataFrame(values, [:Estimate, :SE])
    row_labels = ComponentArrays.labels(theta_hat)
    p_vals = p_values(df_values.Estimate, df_values.SE)
    df_values.Row_Label = row_labels
    df_values.P_Value = p_vals
    df_values = select(df_values, :Row_Label, :Estimate, :SE, :P_Value)
    output_path = joinpath("..", "result", model, k)
    mkpath(output_path)
    estimate_output_csv_path = joinpath(output_path, "est.csv")
    CSV.write(estimate_output_csv_path, df_values)
    println("Estimates saved to $estimate_output_csv_path")
    var_output_csv_path = joinpath(output_path, "var.csv")
    CSV.write(var_output_csv_path, Tables.table(Var), writeheader = false)
    println("Var Matrix saved to $var_output_csv_path")
end

function pred_is_correct(pred_df, acc, action_ranges)
    a_hat = pred_df.a_hat
    a = pred_df.a
    a_cat = categorize(a, action_ranges)
    println("acc: $acc")
    preddf_acc = sum(a_cat .== a_hat)/nrow(pred_df)
    println("preddf_acc: $preddf_acc")
    return acc ≈ sum(a_cat .== a_hat)/nrow(pred_df)
end
