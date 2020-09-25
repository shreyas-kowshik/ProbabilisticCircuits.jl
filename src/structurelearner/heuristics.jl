
using LinearAlgebra: diagind
"""
Pick the edge with maximum flow
"""
function count_downflow(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit)
    dec_id = n.data.node_id
    sum(1:size(flows,1)) do i
        count_ones(flows[i, dec_id]) 
    end
end

function count_downflow(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit, c::LogicCircuit)
    grandpa = n.data.node_id
    prime = c.prime.data.node_id
    sub = c.sub.data.node_id
    edge_count = sum(1:size(flows,1)) do i
        count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
    end
end

function downflow_all(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit, c::LogicCircuit)
    grandpa = n.data.node_id
    prime = c.prime.data.node_id
    sub = c.sub.data.node_id
    edge = map(1:size(flows,1)) do i
        digits(Bool, values[i, prime] & values[i, sub] & flows[i, grandpa], base=2, pad=64)
    end
    vcat(edge...)
end

function eFlow(values, flows, candidates::Vector{Tuple{Node, Node}})
    edge2flows = map(candidates) do (or, and)
        count_downflow(values, flows, or, and)
    end
    (max_flow, max_edge_id) = findmax(edge2flows)
    candidates[max_edge_id], max_flow
end

"""
Pick the variable with maximum sum of mutual information
"""
function vMI(values, flows, edge, vars::Vector{Var}, train_x)
    examples_id = downflow_all(values, flows, edge...)[1:num_examples(train_x)]
    sub_matrix = train_x[examples_id, vars]
    (_, mi) = mutual_information(sub_matrix; α=1.0)
    mi[diagind(mi)] .= 0
    scores = dropdims(sum(mi, dims = 1), dims = 1)
    var = vars[argmax(scores)]
    score = maximum(scores)
    var, score
end

"""
Pick the edge randomly
"""
function eRand(candidates::Vector{Tuple{Node, Node}})
    return rand(candidates)
end

"""
Pick the variable randomly
"""
function vRand(vars::Vector{Var})
    lits = collect(Set{Lit}(scope[and]))
    vars =  Var.(intersect(filter(l -> l > 0, lits), - filter(l -> l < 0, lits)))
    return Var(rand(vars))
end

function independenceMI(mat, prime_lits, sub_lits, lit_map)
        score = 0.0
        t0 = Base.time_ns()
        (_, mi) = mutual_information(mat, nothing; α=1.0)
        t1 = Base.time_ns()
        # println("mi cal : $((t1 - t0)/1.0e9)")
        mi[diagind(mi)] .= 0

        # for p in prime_lits
        #     for s in sub_lits
        #         score += mi[lit_map[p], lit_map[s]]
        #     end
        # end
        mapped_primes = [lit_map[p] for p in prime_lits]
        mapped_subs = [lit_map[s] for s in sub_lits]
        score = sum(mi[mapped_primes, mapped_subs])

        return score
end

function ind_prime_sub(values, flows, candidates::Vector{Tuple{Node, Node}}, scope, data_matrix)
    min_score = Inf
    or0 = nothing
    and0 = nothing
    var0 = nothing

    for (or, and) in candidates
        lits = collect(Set{Lit}(scope[and]))
        lits = sort(collect(intersect(filter(l -> l > 0, lits), - collect(filter(l -> l < 0, lits)))))
        lit_map = Dict(l => i for (i, l) in enumerate(lits))
        vars = Var.(lits)

        prime_lits = sort([l for l in lits if l in scope[children(and)[1]]])
        sub_lits = sort([l for l in lits if l in scope[children(and)[2]]])
        examples_id = downflow_all(values, flows, or, and)[1:num_examples(data_matrix)]

        stotal = 0.0
        t0 = Base.time_ns()
        stotal = independenceMI(data_matrix[examples_id, vars], prime_lits, sub_lits, lit_map)
        t1 = Base.time_ns()
        # println("First Ind Cal : $((t1 - t0)/1.0e9)")

        if stotal == 0.0
            continue
        end

        for var in lits
            pos_scope = examples_id .& data_matrix[:, var]
            neg_scope = examples_id .& (.!(pos_scope))
            s1 = 0.0
            s2 = 0.0

            t0 = Base.time_ns()
            if sum(pos_scope) > 0
                s1 = independenceMI(data_matrix[pos_scope, vars], prime_lits, sub_lits, lit_map)
            end
            t1 = Base.time_ns()
            # println("Second Ind Cal : $((t1 - t0)/1.0e9)")

            t0 = Base.time_ns()
            if sum(neg_scope) > 0
                s2 = independenceMI(data_matrix[neg_scope, vars], prime_lits, sub_lits, lit_map)
            end
            t1 = Base.time_ns()
            # println("Third Ind Cal : $((t1 - t0)/1.0e9)")

            s = s1 + s2

            # if stotal != 0 && s == 0
            #     min_score = s
            #     or0 = or
            #     and0 = and
            #     var0 = var
            #     return min_score, Var.(var0), (or0, and0)
            # end

            # println("Score : $s")

            if s < min_score
                min_score = s
                or0 = or
                and0 = and
                var0 = var
            end
        end
    end

    return min_score, Var.(var0), (or0, and0)
end


function ind_loss(circuit::LogicCircuit, train_x)
    candidates, scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)

    score, var, (or, and) = ind_prime_sub(values, flows, candidates, scope, train_x)
    return (or, and), Var(var)
end


function heuristic_loss(circuit::LogicCircuit, train_x; pick_edge="eFlow", pick_var="vMI")
    candidates, scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)
    if pick_edge == "eFlow"
        edge, flow = eFlow(values, flows, candidates)
    elseif pick_edge == "eRand"
        edge = eRand(candidates)
    else
        error("Heuristics $pick_edge to pick edge is undefined.")
    end

    or, and = edge
    lits = collect(Set{Lit}(scope[and]))
    vars =  Var.(intersect(filter(l -> l > 0, lits), - filter(l -> l < 0, lits)))

    if pick_var == "vMI"
        var, score = vMI(values, flows, edge, vars, train_x)
    elseif pick_var == "vRand"
        var = vRand(vars)
    else
        error("Heuristics $pick_var to pick variable is undefined.")
    end

    return (or, and), var
end

