using LogicCircuits
using ProbabilisticCircuits
using CUDA
using Statistics
using ArgParse

BASE_PATH = "/media/shreyas/Data/UCLA-Intern/Strudel/logs/"


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--name"
            help = "Name of the dataset"
            arg_type = String
      		required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
println("Starting Structure Learning...")
outdir = string(BASE_PATH, parsed_args["name"])


train_x, valid_x, test_x = twenty_datasets(parse_args["name"])

pc = learn_single_model(train_x, valid_x, test_x;
                   log_opts=Dict("outdir"=>outdir, "save"=>1))