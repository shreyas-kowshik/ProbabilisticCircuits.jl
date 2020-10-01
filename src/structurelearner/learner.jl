export learn_single_model
using LogicCircuits: split_step, struct_learn
# using ProbabilisticCircuits
using Statistics: mean
using Random
using DataFrames
using CSV

"""
Learn faithful structured decomposable circuits
"""

OPTS = (
    DEFAULT_LOG_OPT=Dict("valid_x"=>nothing, "test_x"=>nothing, "outdir"=>"", "save"=>1, "print"=>""),
    DEFAULT_SPECIFIC_OPT=Dict("CPT"=>nothing))
TRAIN_HEADER = ["epoch", "time", "total_time", "circuit_size", "train_ll", "valid_ll", "test_ll", "MI", "var", "primitive"]
EM_HEADER = ["iter", "time", "total_time", "train_ll", "valid_ll", "test_ll"]


function save_as_csv(dict; filename, header=keys(dict))
    table = DataFrame(;[Symbol(x) => dict[x] for x in header]...)
    CSV.write(filename, table; )
    table
end

"""
Initialize dictionary with empty contents
"""
function dict_init(header;length=0)
    results = Dict()
    for x in header
        results[x] = Vector{Union{Any,Missing}}(missing, length)
    end
    results
end

"""
"""
function dict_append!(dst::Dict, src::Dict...)
    for dict in src
        for (k, v) in dict
            push!(dst[k], v)
        end
    end
    dst
end

"""
Initialize log dict
"""
function log_init(;opts, specific_opts=nothing, header=TRAIN_HEADER)
    results = dict_init(header)
    if issomething(specific_opts)
        merge!(results, dict_init(CPT_HEADER))
    end
    results
end

"""
Every log step
"""
function log_per_iter(pc, data, results; opts, vtree=nothing, time=missing, epoch=nothing, save_csv=true,
        var=missing, mi=missing, save_freq=500, savecircuit=true)
    # from kwargs
    if !isdir(opts["outdir"])
        mkpath(opts["outdir"])
    end
    continue_flag = true
    @assert vtree!=nothing
    if isnothing(epoch)
        epoch = length(results["epoch"]) + 1
    end
    if savecircuit
        if epoch == 0 && !isdir(joinpath(opts["outdir"], "progress"))
            mkpath(joinpath(opts["outdir"], "progress"))
        end
    end

    push!(results["epoch"], epoch)
    push!(results["var"], var)
    push!(results["MI"], mi)
    push!(results["time"], time)
    push!(results["circuit_size"], num_nodes(pc))

    
    if epoch % 2 == 0
        push!(results["primitive"], "clone")
    else
        push!(results["primitive"], "split")
    end

    if length(results["total_time"]) == 0
        push!(results["total_time"], time)
    else
        push!(results["total_time"], time + results["total_time"][end])
    end


    println("*****************Results****************")
    # ll
    println("Size : $(num_nodes(pc))")
    train_ll = ll = EVI(pc, opts["train_x"])
    println("Train-LL : $(mean(train_ll))")
    push!(results["train_ll"], mean(train_ll))
    valid_ll = missing
    if issomething(opts["valid_x"])
        valid_ll = EVI(pc, opts["valid_x"])

        # Save best circuit
        if length(results["valid_ll"]) > 0
            if mean(valid_ll) >= maximum(results["valid_ll"])
                save_as_psdd(joinpath(opts["outdir"], "progress/best.psdd"), pc, vtree)
            end
        end

        println("Valid-LL : $(mean(valid_ll))")
        push!(results["valid_ll"], mean(valid_ll))
    else
        push!(results["valid_ll"], missing)
    end
    test_ll = missing

    if issomething(opts["test_x"])
        test_ll = EVI(pc, opts["test_x"])
        println("Test-:: : $(mean(test_ll))")
        push!(results["test_ll"], mean(test_ll))
    else
        push!(results["test_ll"], missing)

    end

    # save to csv, pdf
    if haskey(opts, "save") && (save_csv || (epoch % opts["save"] == 0))
        save_as_csv(results; filename=joinpath(opts["outdir"], "progress.csv"), header=TRAIN_HEADER)

        ys = ["train_ll"]
        if issomething(opts["valid_x"])
            push!(ys, "valid_ll")
        end
        if issomething(opts["test_x"])
            push!(ys, "test_ll")
        end
    end

    # Save circuit
    if epoch % save_freq == 0
        save_as_psdd(joinpath(opts["outdir"], "progress/latest.psdd"), pc, vtree)
    end

    println("****************************************")
end

function learn_single_model(train_x, valid_x, test_x;
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=typemax(Int),
        seed=1337,
        log_opts=Dict("outdir"=>"tmp-expresults", "save"=>1))

    # init
    Random.seed!(seed)
    tic = Base.time_ns()
    # pc, vtree = learn_struct_prob_circuit(train_x)
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    
    results = nothing
    if issomething(log_opts)
        log_opts["train_x"] = train_x
        log_opts["valid_x"] = valid_x
        log_opts["test_x"] = test_x
        results = log_init(opts=log_opts)
        toc = Base.time_ns()
        log_per_iter(pc, train_x, results;
        opts=log_opts, vtree=vtree, epoch=0, time=(toc-tic)/1.0e9)
    end

    

    # structure_update
    # loss(circuit) = heuristic_loss(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var)
    loss_split(circuit) = ind_loss_split(circuit, train_x)
    loss_clone(circuit) = ind_loss_clone(circuit, train_x)

    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=loss_split, depth=depth, sanity_check=sanity_check)
        estimate_parameters(c, train_x; pseudocount=pseudocount)
        return c, missing
    end

    pc_clone_step(circuit) = begin
        c::ProbCircuit = clone_step(circuit; loss=loss_clone, depth=depth, sanity_check=sanity_check)
        estimate_parameters(c, train_x; pseudocount=pseudocount)
        return c, missing
    end
    
    iter = 0
    log_per_iter_inline(circuit) = begin
        if issomething(log_opts)
            toc = Base.time_ns()
            log_per_iter(circuit, train_x, results;
            opts=log_opts, vtree=vtree, epoch=iter, time=(toc-tic)/1.0e9)
        end
        iter += 1
        false
    end

    log_per_iter_inline(pc)
    pc = struct_learn(pc; 
        primitives=[pc_split_step, pc_clone_step], kwargs=Dict(pc_split_step=>(), pc_clone_step=>()),
        maxiter=maxiter, stop=log_per_iter_inline)
end
