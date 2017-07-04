include("DrawNet.jl")
include("../utils/Drawing.jl")
module Drawer
using DrawNet, Drawing
using Distributions, PyPlot
using Knet, ArgParse, JLD
global const modelp = "../pretrained/"
global const datap = "../data/"
global const atype = ( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} )
function constructsketch(points)
  points2D = zeros(2, size(points, 2)-1)
  endidxs = [0]
  x = 0
  y = 0
  for i=1:size(points, 2)-1
    if points[3, i] == 0
      push!(endidxs, i)
    end
    x += Float64(points[1, i])
    y += Float64(points[2, i])
    points2D[1, i] = x
    points2D[2, i] = y
  end
  label = "sampled sketch"
  recognized = false
  key_id = "sampled sketch"
  return Sketch(label, recognized, key_id, points2D, endidxs)
end

function adjust_temp(pipdf, temp)
  #=Assumption: pipdf is normalized=#
  return pipdf ./ temp
end

function get_pi_idx(x, pdf; temp=1.0, greedy::Bool = false)
  if greedy
    maxval, maxind = findmax(pdf)
    return maxind
  end
  println("probability sum $(sum(pdf))")
  pdf = adjust_temp(copy(pdf), temp)
  accumulate = 0
  for i=1:length(pdf)
    accumulate += pdf[i]
    if accumulate >= x
      return i
    end
  end
  info("Error with sampling ensemble")
  return -1
end

function sample_gaussian_2d(mu1, mu2, s1, s2, rho; temp = 1.0, greedy::Bool = false)
  if greedy
    return mu1, mu2
  end
  mean = [mu1, mu2]
  s1 *= temp*temp
  s2 *= temp*temp
  cov = [ s1*s1  rho*s1*s2; rho*s1*s2 s2*s2 ]
  x, y = rand(MvNormal(mean,cov)) #sample from multivariate normal
  return x, y
end


function get_mixparams(output, M::Int)
  #Here I used different ordering for outputs; in practice order doesn't matter
  pnorm = softmax(output[:, 1:M], 2) #normalized distribution probabilities
  mu_x = output[:, M+1:2M]
  mu_y = output[:, 2M+1:3M]
  sigma_x = exp(output[:, 3M+1:4M])
  sigma_y = exp(output[:, 4M+1:5M])
  rho = tanh(output[:, 5M+1:6M])
  qlognorm = softmax(output[:, 6M+1:6M+3], 2) #normalized log probabilities of logits
  return pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qlognorm
end

function sample(model, z; seqlen = 45, temperature = 1.0, greedy_mode::Bool = false)
  #=Samples sequence from pretrained model=#
  M = Int((size(model[:output][1], 2)-3)/6) #number of mixtures
  d_H = size(model[:output][1], 1) #decoder hidden unit size
  z_size = size(model[:z][1], 1) #size of latent vector z
  if z == nothing
    z = randn(1, z_size)
  end
  points = zeros(5, seqlen)
  mixture_params = []
  prev_coords = zeros(1, 5)
  prev_coords[1, 3] = 1
  prev_coords = atype(prev_coords)
  greedy = greedy_mode
  temp = temperature
  hc = tanh(z*model[:z][1] .+ model[:z][2])
  state = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
  for i = 1:seqlen
    #dims data = [batchsize, V] = [batchsize, 5]
    input = hcat(prev_coords, z) #concatenate latent vector with previous point
    #input = input * model[:embed]
    state = lstm_lnorm(model[:decode], state, input, model[:dec_shifts][1], model[:dec_shifts][2])
    output =  predict(model[:output], state[1]) #get output params
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm = get_mixparams(output, M) #get mixture parameters and normalized logit values
    idx = get_pi_idx(rand(), pnorm; temp=temp, greedy=greedy)
    idx_eos = get_pi_idx(rand(), qnorm; temp=temp, greedy=greedy)
    eos = [0 0 0]
    eos[idx_eos] = 1
    next_x, next_y = sample_gaussian_2d(mu_x[idx], mu_y[idx], sigma_x[idx], sigma_y[idx], rho[idx]; temp = sqrt(temp), greedy=greedy)
    push!(mixture_params, [ pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm ])
    cur_coords = [next_x next_y eos[1] eos[2] eos[3]]
    points[:, i] = cur_coords'
    prev_coords = atype(copy(cur_coords))
  end
  return points, mixture_params
end

function constructsketchwrong(points)
  @assert(size(points, 1) == 5, "Needs (5, seqlen) matrices")
  endidxs = find(points[3, :].==0) #get indexes of zero elements
  println(size(endidxs))
  endidxs = vcat([0], endidxs)
  label = "sampled sketch"
  recognized = false
  key_id = "sampled sketch"
  return Sketch(label, recognized, key_id, points[1:2, :], endidxs[1:end-1])
end

function decode(model, z; draw_mode = true, temperature = 1.0, factor = 0.2, greedy_mode = false)
  max_seq_length = 50
  sampled_points, mixture_params = sample(model, z; seqlen=max_seq_length,temperature=temperature, greedy_mode=greedy_mode)
  sampled_points = clean_points(sampled_points)
  sketch = constructsketch(sampled_points)
  if draw_mode
    savesketch(sketch, "sampled.png")
  end
end

function getrandomsketch(points3D)
  idx = rand(1:length(points3D))
  #idx = 2
  x_5D = to_big_points(points3D[idx]; max_len = 50)
  batch = []
  for i=1:size(x_5D, 2)
    push!(batch, x_5D[:, i]')
  end

  return map(a->convert(atype, a), batch)
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="Sketch sampler from model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--model"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--dataset"; arg_type=String; default="full_simplified_airplane.jld"; help="Name of the dataset")
    ("--T"; arg_type=Float64; default=1.0; help="Temperature.")
    ("--greedy"; action=:store_true; help="is data preprocessed and ready")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  w = load("$(modelp)$(o[:model])")
  model = revconvertmodel(w["model"])
  info("Model was loaded")
  trnpoints3D, vldpoints3D, tstpoints3D = loaddata("$(datap)$(o[:dataset])")
  info("Train, Valid, Test data obtained")
  x = getrandomsketch(tstpoints3D)
  info("Random sketch was obtained")
  z = getlatentvector(model, x)
  info("got latent vector")
  decode(model, z; temperature=o[:T], greedy_mode=o[:greedy])
end
main()
end
