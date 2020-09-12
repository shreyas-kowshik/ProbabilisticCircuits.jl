# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport
@reexport using LogicCircuits

include("Utils/Utils.jl")
@reexport using .Utils

include("abstract_prob_nodes.jl")
include("plain_prob_nodes.jl")
include("structured_prob_nodes.jl")
include("logistic_nodes.jl")
include("param_bit_circuit.jl")
include("parameters.jl")

include("queries/likelihood.jl")
include("queries/marginal_flow.jl")
include("queries/sample.jl")

include("Logistic/Logistic.jl")
@reexport using .Logistic

# include("exp_flows.jl")
# include("queries.jl")
# include("informations.jl")


# include("reasoning/expectation.jl")
# include("reasoning/exp_flow_circuits.jl")

# include("mixtures/shared_prob_nodes.jl")
# include("mixtures/em.jl")

# include("structurelearner/chow_liu_tree.jl")
# include("structurelearner/init.jl")
# include("structurelearner/heuristics.jl")
# include("structurelearner/learner.jl")


include("LoadSave/LoadSave.jl")
@reexport using .LoadSave

end
