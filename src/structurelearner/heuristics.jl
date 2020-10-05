
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
                            pMI_vec, 
                            # storage_arr,
                            num_prime_vars, num_sub_vars)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    out_index = (index_y - 1)*num_prime_vars + index_x

    # α = 1.0 # Smoothing
    # N = size(prime_mat)[1]
    # num_prime_vars = size(prime_mat)[2]
    # num_sub_vars = size(sub_mat)[2]

    if (index_x > num_prime_vars) || (index_y > num_sub_vars)
        return nothing
    end

#     # prime_var = prime_mat[:, index_x]
#     # sub_var = sub_mat[:, index_y]
#     # not_prime_var = not_prime_mat[:, index_x]
#     # not_sub_var = not_sub_mat[:, index_y]
#     # storage_mat[:, 12] = prime_mat[:, (blockIdx().x - 1) * blockDim().x + threadIdx().x]
#     # storage_mat[:, 13] = sub_mat[:, (blockIdx().y - 1) * blockDim().y + threadIdx().y]
#     # storage_mat[:, 14] = not_prime_mat[:, (blockIdx().x - 1) * blockDim().x + threadIdx().x]
#     # storage_mat[:, 15] = not_sub_mat[:, (blockIdx().y - 1) * blockDim().y + threadIdx().y]

# #     sum_p = sum(prime_var)
#     # sum_s = sum(sub_var)
# #     storage_arr[1] = sum(prime_mat[:, index_x])
#     storage_arr[out_index, 1] = 0.0
#     for k in 1:size(prime_mat)[1]
#         storage_arr[out_index, 1] += prime_mat[k, index_x]
#     end
#     # storage_arr[2] = sum(sub_mat[:, index_y])
#     storage_arr[out_index, 2] = 0.0
#     for k in 1:size(sub_mat)[1]
#         storage_arr[out_index, 2] += sub_mat[k, index_y]
#     end

# #     px = (sum_p + 2.0 * α) / (N + 4.0 * α)
#     # # py = (sum_s + 2.0 * α) / (N + 4.0 * α)
#     # # p_notx = (N - sum_p + 2.0 * α) / (N + 4.0 * α)
#     # # p_noty = (N - sum_s + 2.0 * α) / (N + 4.0 * α)
#     storage_arr[out_index, 3] = (storage_arr[out_index, 1] + 2.0 * α) / (N + 4.0 * α)
#     storage_arr[out_index, 4] = (storage_arr[out_index, 2] + 2.0 * α) / (N + 4.0 * α)
#     storage_arr[out_index, 5] = (N - storage_arr[out_index, 1] + 2.0 * α) / (N + 4.0 * α)
#     storage_arr[out_index, 6] = (N - storage_arr[out_index, 2] + 2.0 * α) / (N + 4.0 * α)
    
    
#     # # p_x_y = (prime_var' * sub_var + α) / (N + 4.0 * α)
#     # # p_notx_y = (not_prime_var' * sub_var + α) / (N + 4.0 * α)
#     # # p_x_noty = (prime_var' * not_sub_var + α) / (N + 4.0 * α)
#     # # p_notx_noty = (not_prime_var' * not_sub_var + α) / (N + 4.0 * α)
#     # # storage_arr[7] = (storage_arr[12]' * storage_arr[13] + α) / (N + 4.0 * α)
#     # # storage_arr[8] = (storage_arr[14]' * storage_arr[13] + α) / (N + 4.0 * α)
#     # # storage_arr[9] = (storage_arr[12]' * storage_arr[15] + α) / (N + 4.0 * α)
#     # # storage_arr[10] = (storage_arr[14]' * storage_arr[15] + α) / (N + 4.0 * α)
#     storage_arr[out_index, 7] = 0.0
#     for k in 1:N
#         storage_arr[out_index, 7] += (prime_mat[k, index_x] & sub_mat[k, index_y])
#     end
#     storage_arr[out_index, 7] = (storage_arr[out_index, 7] + α) / (N + 4.0 * α)
    
#     storage_arr[out_index, 8] = 0.0
#     for k in 1:N
#         storage_arr[out_index, 8] += (not_prime_mat[k, index_x] & sub_mat[k, index_y])
#     end
#     storage_arr[out_index, 8] = (storage_arr[out_index, 8] + α) / (N + 4.0 * α)
    
#     storage_arr[out_index, 9] = 0.0
#     for k in 1:N
#         storage_arr[out_index, 9] += (prime_mat[k, index_x] & not_sub_mat[k, index_y])
#     end
#     storage_arr[out_index, 9] = (storage_arr[out_index, 9] + α) / (N + 4.0 * α)
    
#     storage_arr[out_index, 10] = 0.0
#     for k in 1:N
#         storage_arr[out_index, 10] += (not_prime_mat[k, index_x] & not_sub_mat[k, index_y])
#     end
#     storage_arr[out_index, 10] = (storage_arr[out_index, 10] + α) / (N + 4.0 * α)
    
#     # storage_arr[7] = (prime_mat[:, index_x]' * sub_mat[:, index_y] + α) / (N + 4.0 * α)
#     # storage_arr[8] = (not_prime_mat[:, index_x]' * sub_mat[:, index_y] + α) / (N + 4.0 * α)
#     # storage_arr[9] = (prime_mat[:, index_x]' * not_sub_mat[:, index_y] + α) / (N + 4.0 * α)
#     # storage_arr[10] = (not_prime_mat[:, index_x]' * not_sub_mat[:, index_y] + α) / (N + 4.0 * α)

#     storage_arr[out_index, 11] = 0.0
      # storage_arr[11] += (p_x_y * CUDA.log((p_x_y)/(p_x * p_y)))
      # storage_arr[11] += (p_notx_y * CUDA.log((p_notx_y)/(p_notx * p_y)))
      # storage_arr[11] += (p_x_noty * CUDA.log((p_x_noty)/(p_x * p_noty)))
      # storage_arr[11] += (p_notx_noty * CUDA.log((p_notx_noty)/(p_notx * p_noty)))
#     storage_arr[out_index, 11] += (storage_arr[out_index, 7] * CUDA.log((storage_arr[out_index, 7])/(storage_arr[out_index, 3] * storage_arr[out_index, 4] + 1e-6) + 1e-6))
#     storage_arr[out_index, 11] += (storage_arr[out_index, 8] * CUDA.log((storage_arr[out_index, 8])/(storage_arr[out_index, 5] * storage_arr[out_index, 4] + 1e-6) + 1e-6))
#     storage_arr[out_index, 11] += (storage_arr[out_index, 9] * CUDA.log((storage_arr[out_index, 9])/(storage_arr[out_index, 3] * storage_arr[out_index, 6] + 1e-6) + 1e-6))
#     storage_arr[out_index, 11] += (storage_arr[out_index, 10] * CUDA.log((storage_arr[out_index, 10])/(storage_arr[out_index, 5] * storage_arr[out_index, 6] + 1e-6) + 1e-6))
    
    pMI_vec[index_x, index_y] = 0.0

    if index_x == index_y
        return nothing
    end

    pMI_vec[index_x, index_y] += (p_s[index_x, index_y] * CUDA.log((p_s[index_x, index_y])/(marginals[index_x] * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_s[index_x, index_y] * CUDA.log((notp_s[index_x, index_y])/((1.0 - marginals[index_x]) * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (p_nots[index_x, index_y] * CUDA.log((p_nots[index_x, index_y])/(marginals[index_x] * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_nots[index_x, index_y] * CUDA.log((notp_nots[index_x, index_y])/((1.0 - marginals[index_x]) * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))

    # pMI_vec[index_x, index_y] += (p_s[index_x, index_y] * CUDA.log((p_s[index_x, index_y])/(p_s[index_x, index_x] * p_s[index_y, index_y])))
    # pMI_vec[index_x, index_y] += (notp_s[index_x, index_y] * CUDA.log((notp_s[index_x, index_y])/((1.0 - p_s[index_x, index_x]) * p_s[index_y, index_y])))
    # pMI_vec[index_x, index_y] += (p_nots[index_x, index_y] * CUDA.log((p_nots[index_x, index_y])/(p_s[index_x, index_x] * (1.0 - p_s[index_y, index_y]))))
    # pMI_vec[index_x, index_y] += (notp_nots[index_x, index_y] * CUDA.log((notp_nots[index_x, index_y])/((1.0 - p_s[index_x, index_x]) * (1.0 - p_s[index_y, index_y]))))
    # pMI_vec[out_index] += (p_notx_y * CUDA.log((p_notx_y)/(p_notx * p_y)))
    # pMI_vec[out_index] += (p_x_noty * CUDA.log((p_x_noty)/(p_x * p_noty)))
    # pMI_vec[out_index] += (p_notx_noty * CUDA.log((p_notx_noty)/(p_notx * p_noty)))


    # pMI_vec[out_index] = storage_arr[out_index, 11]
#     pMI_vec[(index_y - 1)*num_sub_vars + index_x] = 0.0
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

    # t0 = Base.time_ns()
    # d_d = dmat' * dmat
    # d_nd = dmat' * (.!(dmat))
    # nd_nd = (.!(dmat))' * (.!(dmat))
    # t1 = Base.time_ns()
    # println("T0 : $((t1 - t0)/1.0e9)")

    # t0 = Base.time_ns()
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



    ########## DEBUG ############
    # println(d_d)
    # println(d_nd)
    # println(nd_nd)
    # println(N)

    #############################


    # t1 = Base.time_ns()
    # println("T1 : $((t1 - t0)/1.0e9)")

    d_d = (d_d .+ (4.0 * α)) ./ (N + 4.0 * α)
    d_nd = (d_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    nd_nd = (nd_nd .+ (4.0 * α)) ./ (N + 4.0 * α)
    marginals = (dropdims(count(dmat, dims=1), dims=1) .+ (2.0 * α)) ./ (N + 4.0 * α)

    # println(d_d)
    # println(d_nd)
    # println(nd_nd)
    # println(N)
    # println(marginals)

    # p_marginal = marginals[Var.(mapped_primes)]
    # s_marginal = marginals[Var.(mapped_subs)]

    # println("P_Marginal : $(d_d[1, 1])")
    # println("S_Marginal : $(d_d[2, 2])")

    # error("debug")

    p_s = d_d
    p_nots = d_nd
    notp_s = collect(d_nd')
    notp_nots = nd_nd








    # Data Type Conversions #
    # prime_gpu = to_gpu(convert(Matrix, prime_mat))
    # println("PrimeGPU : $(sum(prime_gpu)) :: $(size(prime_mat)) :: $(typeof(prime_mat))")
    # sub_gpu = to_gpu(convert(Matrix, sub_mat))
    # println("PrimeGPU : $(sum(prime_gpu)) :: $(size(prime_gpu)) :: $(typeof(prime_gpu))")
    # println("***********************")
    # not_prime_gpu = to_gpu(convert(Matrix, .!(prime_mat)))
    # not_sub_gpu = to_gpu(convert(Matrix, .!(sub_mat)))
    # storage_arr = to_gpu(Array{Float64}(undef, num_prime_vars*num_sub_vars, 15))

    @cuda threads=num_threads blocks=num_blocks independenceMI_gpu(to_gpu(marginals),
                                                p_s, to_gpu(notp_s), p_nots, notp_nots,
                                                pMI_vec,
                                                # storage_arr, 
                                                num_vars, num_vars)

    
    # println(pMI_vec)
    # cpu_pMI = CUDA.sum(pMI_vec)
    cpu_pMI = to_cpu(pMI_vec)
    # println("Vars : $(num_prime_vars + num_sub_vars)")
    # println(cpu_pMI)
    cpu_pMI = cpu_pMI[Var.(mapped_primes), Var.(mapped_subs)]
    # println(cpu_pMI)
    cpu_pMI = sum(cpu_pMI)
    # println(cpu_pMI)
    # println("-*-*-*-*-*-")

    if abs(cpu_pMI) < 1e-6
        cpu_pMI = 0.0
    end

    return cpu_pMI
end

function independenceMI(mat, marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
        # score = 0.0
        # t0 = Base.time_ns()
        # (_, mi) = mutual_information(mat, nothing; α=1.0)
        # t1 = Base.time_ns()
        # # println("mi cal : $((t1 - t0)/1.0e9)")
        # mi[diagind(mi)] .= 0

        # # for p in prime_lits
        # #     for s in sub_lits
        # #         score += mi[lit_map[p], lit_map[s]]
        # #     end
        # # end
        # mapped_primes = [lit_map[p] for p in prime_lits]
        # mapped_subs = [lit_map[s] for s in sub_lits]
        # score = sum(mi[mapped_primes, mapped_subs])

        # return score
        return independenceMI_gpu_wrapper(mat, marginals, d_d, d_nd, nd_nd, prime_lits, sub_lits, lit_map)
end


function s_min(s1, s2)
    s = 0.0
    if s1 == Inf
        s = s2
    elseif s2 == Inf
        s = s1
    else
        s = min(s1,s2)
    end
    return s
end

function s_weighted(s1, s2)
    s = 0.0
    if s1 == Inf
        s = s2
    elseif s2 == Inf
        s = s1
    else
        w1 = s1 / (s1 + s2)
        w2 = s2 / (s1 + s2)
        s = w1*s1 + s2*s2
    end
    return s
end

function s_max(s1, s2)
    s = 0.0
    if s1 == Inf
        s = s2
    elseif s2 == Inf
        s = s1
    else
        s = max(s1, s2)
    end
    return s
end



function ind_prime_sub(pc, values, flows, candidates::Vector{Tuple{Node, Node}}, scope, data_matrix)
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
    min_s1 = Inf
    min_s2 = Inf
    min_s = Inf
    num_vars = nothing
    or0 = nothing
    and0 = nothing
    var0 = nothing


    # Choose one layer to reduce computation #
    bc = BitCircuit(pc, data_matrix)
    id2layer = Dict()
    cands = []
    for (i, layer) in enumerate(bc.layers)
        for id in layer
            id2layer[id] = i
        end
    end

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

    for layer_id in 1:length(layered_cands)
        if done == true
            break
        end

        candidates = layered_cands[layer_id]

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
                # println("Already faithful")
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
                # if s1 == Inf
                #     s = s2
                # elseif s2 == Inf
                #     s = s1
                # else
                #     s = s1 + s2
                # end
                s = s_max(s1, s2)

                t1 = Base.time_ns()
                # println("i:$i / $(length(candidates)), One Check Time : $((t1 - t0)/1.0e9)")

                # println("i: $i, var:$var, pos_scope : $(sum(pos_scope)), neg_scope : $(sum(neg_scope)), stotal : $stotal, s : $s")

                # res[j] = s

                # println("S : $s")
                # println("stotal : $stotal")

		        # s = s - stotal

                num_vars = length(prime_sub_lits)
                s = s / (1.0 * num_vars)

                

                if s < min_score
                    min_score = s
                    or0 = or
                    and0 = and
                    var0 = var
                    done=true

                    min_s1 = s1 / (1.0 * num_vars)
                    min_s2 = s2 / (1.0 * num_vars)
                    min_s = stotal / (1.0 * num_vars)

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
        return -1, nothing, nothing, nothing
    end

    return [min_score, min_s1, min_s2, min_s, num_vars, layer_used] , Var.(var0), (or0, and0)
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

function ind_loss_split(circuit::LogicCircuit, train_x)
    candidates, scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)

    info_arr, var, (or, and) = ind_prime_sub(circuit, values, flows, candidates, scope, train_x)

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

