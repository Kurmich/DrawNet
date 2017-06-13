module DrawNet
include("Drawing.jl")
include("DataLoader.jl")
using Drawing, DataLoader
using Knet, ArgParse, JLD

function initfullmodel(model, H, V; atype=( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} ) )
  #initial hidden and cell states of forward encoder
end

function initencoder(model, H, V; atype=( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} ))
  init(d...) = atype(xavier(d...))
  model[:fw_state0] = [init(1, H), init(1, H)]
  model[:fw_embed] = init(V, H)
  model[:fw_encode] = [ init(2H, 4H), init(1, 4H) ]
  model[:bw_state0] = [init(1, H), init(1, H)]
  model[:bw_embed] = init(V, H)
  model[:bw_encode] = [ init(2H, 4H), init(1, 4H) ]
end

function initpredecoder(model, H, z_size, ; atype=( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} ))
  init(d...) = atype(xavier(d...))
  model[:mu] = [init(), init()]
  model[:sigma] = [init(), init()]
  model[:z] = [init(), init()]
end

function initdecoder(model, H, V; atype=( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} ))
  model[:embed] = []
  model[:decode] = []
  model[:output] = []
end

function lstm(param, state, input)
  weight, bias = param
  hidden, cell = state
  h = size(hidden, 2)
  gates = hcat(input, hidden) * weight .+ bias
  forget = sigm(gates[:, 1:h])
  ingate = sigm(gates[:, 1+h:2h])
  outgate = sigm(gates[:, 1+2h:3h])
  change = tanh(gates[:, 1+3h:4h])
  cell = cell .* forget + ingate .* change
  hidden = outgate .* tanh(cell)
  return (hidden, cell)
end


function main(args=ARGS)
  s = ArgParseSettings()
  s.description="My Model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--numsteps"; arg_type=Int; default=100000; help="Total number of training set. Keep large.")
    ("--save_every"; arg_type=Int; default=500; help="Number of batches per checkpoint creation.")
    ("--dec_model"; arg_type=String; default="lstm"; help="Decoder: lstm, or ....")
    ("--dec_rnn_size"; arg_type=Int; default=512; help="Size of decoder.")
    ("--enc_model"; arg_type=String; default="lstm"; help="Ecoder: lstm, or ....")
    ("--enc_rnn_size"; arg_type=Int; default=256; help="Size of encoder.")
    ("--batchsize"; arg_type=Int; default=100; help="Minibatch size. Recommend leaving at 100.")
    ("--grad_clip"; arg_type=Float64; default=1.0; help="Gradient clipping. Recommend leaving at 1.0.")
    ("--num_mixture"; arg_type=Int; default=20; help="Number of mixtures in Gaussian mixture model.")
    ("--z_size"; arg_type=Int; default=128; help="Size of latent vector z. Recommend 32, 64 or 128.")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  model = Dict{Symbol, Any}()

end

end
