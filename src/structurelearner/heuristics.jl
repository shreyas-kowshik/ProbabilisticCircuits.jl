
using LinearAlgebra: diagind
using CUDA: CUDA, @cuda
using DataFrames: DataFrame

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

function clone_downflow_all(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit, a1::LogicCircuit, a2::LogicCircuit)
    or = n.data.node_id
    p1 = a1.prime.data.node_id
    s1 = a1.sub.data.node_id
    p2 = a2.prime.data.node_id
    s2 = a2.sub.data.node_id

    edge1 = map(1:size(flows,1)) do i
        and1_downflow = (values[i, p1] & values[i, s1] & flows[i, or])
        # and2_downflow = (values[i, p2] & values[i, s2] & flows[i, or])
        # or_rem = flows[i, or] & (!and2_downflow) & (!and2_downflow)

        digits(Bool, and1_downflow, base=2, pad=64)
    end

    edge2 = map(1:size(flows,1)) do i
        # and1_downflow = (values[i, p1] & values[i, s1] & flows[i, or])
        and2_downflow = (values[i, p2] & values[i, s2] & flows[i, or])
        # or_rem = flows[i, or] & (!and2_downflow) & (!and2_downflow)

        digits(Bool, and2_downflow, base=2, pad=64)
    end

    edge3 = map(1:size(flows,1)) do i
        # and1_downflow = (values[i, p1] & values[i, s1] & flows[i, or])
        # and2_downflow = (values[i, p2] & values[i, s2] & flows[i, or])
        # or_rem = flows[i, or] & (!and2_downflow) & (!and2_downflow)

        digits(Bool, flows[i, or], base=2, pad=64)
    end

    vcat(edge1...), vcat(edge2...), vcat(edge3...)
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

function independenceMI_gpu(marginals, p_s, notp_s, p_nots, notp_nots, 
                            pMI_vec, num_prime_vars, num_sub_vars)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (index_x > num_prime_vars) || (index_y > num_sub_vars)
        return nothing
    end

    pMI_vec[index_x, index_y] = 0.0

    if index_x == index_y
        return nothing
    end

    pMI_vec[index_x, index_y] += (p_s[index_x, index_y] * CUDA.log((p_s[index_x, index_y])/(marginals[index_x] * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_s[index_x, index_y] * CUDA.log((notp_s[index_x, index_y])/((1.0 - marginals[index_x]) * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (p_nots[index_x, index_y] * CUDA.log((p_nots[index_x, index_y])/(marginals[index_x] * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_nots[index_x, index_y] * CUDA.log((notp_nots[index_x, index_y])/((1.0 - marginals[index_x]) * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))

    return nothing
end

function independenceMI_gpu_wrapper(dmat, marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
    mapped_primes = [lit_map[p] for p in prime_lits]
    mapped_subs = [lit_map[s] for s in sub_lits]

    num_prime_vars = length(mapped_primes)
    num_sub_vars = length(mapped_subs)
    num_vars = num_prime_vars + num_sub_vars

    pMI_vec = to_gpu(zeros(num_vars, num_vars))

    num_threads = (16, 16)
    num_blocks = (ceil(Int, num_vars/16), ceil(Int, num_vars/16))

    α = 0.0
    N = size(dmat)[1]

    dummy = ones(num_prime_vars+num_sub_vars,num_prime_vars+num_sub_vars)
    d_d = cu(similar(dummy))
    d_nd = cu(similar(dummy))
    nd_nd = cu(similar(dummy))
    dmat_gpu = cu(dmat)
    dmat_tr_gpu = cu(collect(dmat'))
    not_dmat_gpu = cu(.!(dmat))
    not_dmat_tr_gpu = cu(collect((.!(dmat))'))

    mul!(d_d, dmat_tr_gpu, dmat_gpu)
    mul!(d_nd, dmat_tr_gpu, not_dmat_gpu)
    mul!(nd_nd, not_dmat_tr_gpu, not_dmat_gpu)

    d_d = (d_d .+ (4.0 * α)) ./ (N + 4.0 * α)
    d_nd = (d_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    nd_nd = (nd_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    marginals = (dropdims(count(dmat, dims=1), dims=1) .+ (2.0 * α)) ./ (N + 4.0 * α)

    p_s = d_d
    p_nots = d_nd
    notp_s = collect(d_nd')
    notp_nots = nd_nd

    @cuda threads=num_threads blocks=num_blocks independenceMI_gpu(to_gpu(marginals),
                                                p_s, to_gpu(notp_s), p_nots, notp_nots,
                                                pMI_vec, num_vars, num_vars)

    
    cpu_pMI = to_cpu(pMI_vec)
    cpu_pMI = cpu_pMI[Var.(mapped_primes), Var.(mapped_subs)]
    cpu_pMI = mean(cpu_pMI)

    if abs(cpu_pMI) < 1e-10
        cpu_pMI = 0.0
    end

    return cpu_pMI
end

function ind_prime_sub(pc, values, flows, candidates::Vector{Tuple{Node, Node}}, scope, data_matrix;
                       iter=iter)
    dmat = BitArray(convert(Matrix, data_matrix))
    d_d = dmat' * dmat
    d_nd = dmat' * (.!(dmat))
    nd_nd = (.!(dmat))' * (.!(dmat))

    N = size(data_matrix)[1]
    α = 0.0

    d_d = (d_d .+ (4.0 * α)) ./ (N + 4.0 * α)
    d_nd = (d_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    nd_nd = (nd_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    marginals = (dropdims(count(dmat, dims=1), dims=1) .+ (2.0 * α)) ./ (N + 4.0 * α)

    min_score = Inf
    min_s1 = Inf
    min_s2 = Inf
    min_s = Inf
    min_size_s = Inf
    num_vars = nothing
    or0 = nothing
    and0 = nothing
    var0 = nothing

    α_size = 0.0
    overall_size = num_nodes(pc)


    # Choose one layer to reduce computation #
    bc = BitCircuit(pc, data_matrix)
    id2layer = Dict()
    cands = []
    for (i, layer) in enumerate(bc.layers)
        for id in layer
            id2layer[id] = i
        end
    end

    # println("LAYERS : $(length(bc.layers))")

    layered_cands = Vector{Vector{Tuple{Node, Node}}}(undef, length(bc.layers))
    for i in 1:length(layered_cands)
        layered_cands[i] = []
    end

    for (or, and) in candidates
        push!(layered_cands[id2layer[or.data.node_id]], (or, and))
    end

    # candidates = rand(layered_cands)
    # println(layered_cands)
    layered_cands = filter(x->length(x) > 0, layered_cands)
    # println(length(layered_cands))
    # layer_id = 1
    
    # candidates = rand(layered_cands)
    ##########################################


    tvar0 = Base.time_ns()

    done = false
    layer_used = -1

    sum_ex = -1
    sum_pos = -1
    sum_neg = -1

    min_score1 = Inf
    or01 = -1
    and01 = -1
    var01 = -1
    min_s11 = -1
    min_s21 = -1
    min_s1 = -1
    min_size_s1 = -1

    sum_ex1 = -1
    sum_pos1 = -1
    sum_neg1 = -1

    num_vars1 = -1
    layer_used1 = -1

    # layer_id = 
    # println(iter)
    # # iter = iter + 1
    # layer_map = Dict()
    # layer_map[length(layered_cands)] = 1

    # for t1 in length(layered_cands)-1:-1:1
    #     layer_map[t1] = minimum([layer_map[t1+1]*2, 64])
    # end

    # println(layer_map)
    # # println(collect(values(layer_map)))

    # # t1 = sum(values(layer_map))
    # t1 = 0.0
    # for (k,v) in layer_map
    #     t1 += v
    # end

    # if t1 <= 10000
    #     iter = ((iter - 1)%t1) + 1
    # end

    # tup1 = sort(collect(layer_map), by=x->x[2])
    # println(tup1)
    # for i in 2:length(tup1)
    #     tup1[i] = (tup1[i][1] => tup1[i - 1][2] + tup1[i][2])
    # end

    # println(tup1)

    # for i in 1:length(tup1)-1
    #     if iter > tup1[i][2] && iter <= tup1[i+1][2]
    #         lid = tup1[i+1][1]
    #         break
    #     end
    # end

    # if lid == -1
    #     lid = 1
    # end

    # total_count = 0

    for layer_id in 1:1
    # for layer_id in 1:layered_cands
    # while true
    #     total_count += 1

    #     if total_count > 1
    #         lid = lid + 1
    #         lid = ((lid - 1)%length(layered_cands)) + 1
    #     end

    #     # layer_id = length(layered_cands) - lid + 1
    #     layer_id = lid
    #     # println("layer_id : $layer_id, length : $(length(layered_cands))")

        if done == true
            break
        end

        # layer_id = 1
        # candidates = layered_cands[layer_id]
        # println(length(candidates))
        # println(layer_id)


        for (i, (or, and)) in enumerate(candidates)
            og_lits = collect(Set{Lit}(scope[and])) # All literals

            # On which you can split
            lits = sort(collect(intersect(filter(l -> l > 0, og_lits), - collect(filter(l -> l < 0, og_lits)))))
            vars = Var.(lits)

            prime_lits = sort([abs(l) for l in og_lits if l in scope[children(and)[1]]])
            sub_lits = sort([abs(l) for l in og_lits if l in scope[children(and)[2]]])

            prime_lits = sort(collect(Set{Lit}(prime_lits)))
            sub_lits = sort(collect(Set{Lit}(sub_lits)))
            prime_sub_lits = sort([prime_lits..., sub_lits...])

            # println("Length of candidates : $(length(candidates)), layer_id : $layer_id, scope : $(length(prime_sub_lits))")
            
            @assert length(prime_lits) > 0 "Prime litset empty"
            @assert length(sub_lits) > 0 "Sub litset empty"
            prime_sub_vars = Var.(prime_sub_lits)
            lit_map = Dict(l => i for (i, l) in enumerate(prime_sub_lits))

            examples_id = downflow_all(values, flows, or, and)[1:num_examples(data_matrix)]

            if(sum(examples_id) == 0)
                continue
            end

            stotal = 0.0

            # println("---stotal---")
            stotal = independenceMI_gpu_wrapper(dmat[examples_id, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)

            if stotal == 0.0
                # println("layer_id:$layer_id, Already faithful!!!\n\n\n")
                continue
            end

            if(length(lits) == 0)
                # println("No Variales")
                continue
            end

            res = zeros(length(lits))

            
            # T = Threads.nthreads()
            # println("Threads : $T")

            # Threads.@threads for t=1:T
            # println("t : $t :: $(length(lits))")
            for j=1:length(lits)
                # println("t : $t, j : $j :: $(length(lits))")
                var = lits[j]
                t0 = Base.time_ns()
                pos_scope = examples_id .& data_matrix[:, var]
                neg_scope = examples_id .& (.!(pos_scope))
                @assert sum(examples_id) == (sum(pos_scope) + sum(neg_scope)) "Scopes do not add up"
                s1 = Inf
                s2 = Inf

                if sum(pos_scope) > 0
                    # println("---s1---")
                    s1 = independenceMI_gpu_wrapper(dmat[pos_scope, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
                end
                if sum(neg_scope) > 0
                    # println("---s2---")
                    s2 = independenceMI_gpu_wrapper(dmat[neg_scope, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
                end

                s = 0.0
                w1 = (sum(pos_scope)) / (1.0 * N)
                w2 = (sum(neg_scope)) / (1.0 * N)
                w = (sum(examples_id)) / (1.0 * N)

                if s1 == Inf
                    s = (s2*w2) - (stotal*w)
                elseif s2 == Inf
                    s = (s1*w1) - (stotal*w)
                else
                    s = (s1*w1) + (s2*w2) - (2.0*stotal*w)
                end

                t1 = Base.time_ns()
                # println("i:$i / $(length(candidates)), One Check Time : $((t1 - t0)/1.0e9)")

                # println("i: $i, var:$var, pos_scope : $(sum(pos_scope)), neg_scope : $(sum(neg_scope)), stotal : $stotal, s : $s")

                # res[j] = s

                # println("S1 : $s1")
                # println("S2 : $s2")
                # println("stotal : $stotal")

		        # s = s - stotal

                num_vars = length(prime_sub_lits)
                # s = s / (1.0 * num_vars)

                size_score = (num_nodes(or)) / (1.0 * overall_size)
                # size_score = num_nodes(or) * 0.0001
                # s += (α_size * size_score)
                s = (10.0 * s) / size_score
                # s = (1000.0 * s)



                if s < min_score1
                    min_score1 = s
                    or01 = or
                    and01 = and
                    var01 = var
                    # done=true

                    min_s11 = s1
                    min_s21 = s2
                    min_s1 = stotal
                    min_size_s1 = size_score
                    num_vars1 = length(prime_sub_lits)
                    sum_ex1 = sum(examples_id)
                    sum_pos1 = sum(pos_scope)
                    sum_neg1 = sum(neg_scope)


                    layer_used1 = layer_id
                end

                if w1 < 0.1 && s1 != Inf
                    continue
                end

                if w2 < 0.1 && s2 != Inf
                    continue
                end

                # if w < 0.05
                #     continue
                # end

                




                if s < min_score
                    min_score = s
                    or0 = or
                    and0 = and
                    var0 = var
                    done=true

                    min_s1 = s1
                    min_s2 = s2
                    min_s = stotal
                    min_size_s = size_score
                    num_vars = length(prime_sub_lits)
                    sum_ex = sum(examples_id)
                    sum_pos = sum(pos_scope)
                    sum_neg = sum(neg_scope)

                    layer_used = layer_id
                end
            end


            # idx = argmin(res)
            # s = res[idx]
            # var = lits[idx]

            # if s < min_score
            #     min_score = s
            #     or0 = or
            #     and0 = and
            #     var0 = var
            # end

        end
    end

    tvar1 = Base.time_ns()
    println("Looping Time : $((tvar1 - tvar0)/1.0e9)")

    if or0 == nothing || and0 == nothing || var0 == nothing
        println("W1 condition not satisfied by any candidate!")
        min_score = min_score1
        or0 = or01
        and0 = and01
        var0 = var01

        min_s1 = min_s11
        min_s2 = min_s21
        min_s = min_s1
        min_size_s = min_size_s1
        num_vars = num_vars1
        sum_ex = sum_ex1
        sum_pos = sum_pos1
        sum_neg = sum_neg1

        layer_used = layer_used1
    end

    println("Min Size S : $(min_size_s)")
    println("Min H : $(min_score - min_size_s)")


    if or0 == nothing || and0 == nothing || var0 == nothing   
        return -1, nothing, nothing, nothing
    end

    return [min_score, min_s1, min_s2, min_s, min_size_s, num_vars, layer_used, sum_ex, sum_pos, sum_neg] , Var.(var0), (or0, and0)
end

function ind_clone(values, flows, candidates::Vector{Tuple{Node, Node, Node}}, scope, data_matrix)
    dmat = BitArray(convert(Matrix, data_matrix))
    d_d = dmat' * dmat
    d_nd = dmat' * (.!(dmat))
    nd_nd = (.!(dmat))' * (.!(dmat))

    N = size(data_matrix)[1]
    α = 1.0
    d_d = (d_d .+ (4.0 * α)) ./ (N + 4.0 * α)
    d_nd = (d_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    nd_nd = (nd_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    marginals = (dropdims(count(dmat, dims=1), dims=1) .+ (2.0 * α)) ./ (N + 4.0 * α)

    min_score = Inf
    or0 = nothing
    and00 = nothing
    and01 = nothing

    for (i, (or, and1, and2)) in enumerate(candidates)
        check_arr = [is⋀gate(ch) for ch in children(or)]
        if sum(check_arr) == 0
            continue
        end

        and_child = children(or)[1]
        og_lits = collect(Set{Lit}(scope[and_child])) # All literals
        # On which you can split
        lits = sort(collect(intersect(filter(l -> l > 0, og_lits), - collect(filter(l -> l < 0, og_lits)))))
        # lit_map = Dict(l => i for (i, l) in enumerate(lits))
        vars = Var.(lits)

        prime_lits = sort([abs(l) for l in og_lits if l in scope[children(and_child)[1]]])
        sub_lits = sort([abs(l) for l in og_lits if l in scope[children(and_child)[2]]])
        prime_lits = collect(Set{Lit}(prime_lits))
        sub_lits = collect(Set{Lit}(sub_lits))
        
        prime_sub_lits = sort([prime_lits..., sub_lits...])
        
        @assert length(prime_lits) > 0 "Prime litset empty"
        @assert length(sub_lits) > 0 "Sub litset empty"
        prime_sub_vars = Var.(prime_sub_lits)
        lit_map = Dict(l => i for (i, l) in enumerate(prime_sub_lits))

        and1_downflow, and2_downflow, or_downflow = clone_downflow_all(values, flows, or, and1, and2)
        and1_downflow = and1_downflow[1:num_examples(data_matrix)]
        and2_downflow = and2_downflow[1:num_examples(data_matrix)]
        or_downflow = or_downflow[1:num_examples(data_matrix)]

        examples_id = or_downflow
        or_rem_downflow = or_downflow .& (.!and1_downflow) .& (.!and2_downflow)
        examples_id1 = and1_downflow .| or_rem_downflow
        examples_id2 = and2_downflow .| or_rem_downflow


        # examples_id1 = examples_id1[1:num_examples(data_matrix)]
        # examples_id2 = examples_id2[1:num_examples(data_matrix)]
        # examples_id = examples_id1 | examples_id2

        if(sum(examples_id) == 0)
            continue
        end

        stotal = 0.0
        cur_score = 0.0

        for and in children(or)
            # Check independence scores
            examples_and = map(1:size(flows,1)) do i
                digits(Bool, values[i, and.prime.data.node_id] & values[i, and.sub.data.node_id], base=2, pad=64)
            end
            examples_and = vcat(examples_and...)[1:num_examples(data_matrix)]

            ex_and_before = examples_id .& examples_and
            ex_and1 = examples_id1 .& examples_and
            ex_and2 = examples_id2 .& examples_and

            sbefore = independenceMI_gpu_wrapper(dmat[ex_and_before, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
            s1 = independenceMI_gpu_wrapper(dmat[ex_and1, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
            s2 = independenceMI_gpu_wrapper(dmat[ex_and2, prime_sub_vars], marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)

            stotal += sbefore
            cur_score += (s1 + s2)
        end

        if stotal == 0.0
            continue
        end

	    # cur_score = cur_score - stotal
        if cur_score < min_score
            min_score = cur_score
            or0 = or
            and00 = and1
            and01 = and2
        end
    end

    return min_score, or0, and00, and01
end

function ind_loss_split(circuit::LogicCircuit, train_x; iter=iter)
    candidates, scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)

    info_arr, var, (or, and) = ind_prime_sub(circuit, values, flows, candidates, scope, train_x; iter=iter)

    return info_arr, (or, and), Var(var)
end

function ind_loss_clone(circuit::LogicCircuit, train_x)
    _, scope = split_candidates(circuit)
    candidates = clone_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)

    score, or, and1, and2 = ind_clone(values, flows, candidates, scope, train_x)
    return score, or, and1, and2
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

