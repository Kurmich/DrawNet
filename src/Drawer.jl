include("DrawNet.jl")
include("../utils/Drawing.jl")
module Drawer
using DrawNet, Drawing
using Distributions, PyPlot, SVR
using Knet, ArgParse, JLD
using IDM
include("../rnns/RNN.jl")
include("Ferret.jl")

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

function stroke_constructsketch(points)
  endidxs = [0]
  allpoints = nothing
  for stroke in points
    points2D = zeros(2, size(stroke, 2)-1)
    x = 0
    y = 0
    for i=1:size(stroke, 2)-1
      x += Float64(stroke[1, i])
      y += Float64(stroke[2, i])
      points2D[1, i] = x
      points2D[2, i] = y
      if stroke[3, i] == 0
        push!(endidxs, endidxs[end] + i)
      end
    end
    if allpoints == nothing
      allpoints = points2D
    else
      allpoints = hcat(allpoints, points2D)
    end
  end
  label = "sampled sketch"
  recognized = false
  key_id = "sampled sketch"
  return Sketch(label, recognized, key_id, allpoints, endidxs)
end

function adjust_temp(pipdf, temp)
  #=Assumption: pipdf is normalized?=#
  pipdf = log(pipdf)/temp
  pipdf = pipdf .- maximum(pipdf)
  pipdf = exp(pipdf)
  pipdf = pipdf ./ sum(pipdf, 2)
  return pipdf
end

function get_pi_idx(x, pdf; temp=1.0, greedy::Bool = false)
  if greedy
    maxval, maxind = findmax(pdf)
    return maxind
  end
  #println("probability sum $(sum(pdf))")
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

function sample(model, z; seqlen::Int = 45, temperature = 1.0, greedy_mode::Bool = false)
  #=Samples sequence from pretrained model=#
  M = Int((size(model[:output][1], 2)-3)/6) #number of mixtures
  d_H = size(model[:output][1], 1) #decoder hidden unit size
  z_size = size(model[:z][1], 1) #size of latent vector z
  forgetcells = []
  incells = []
  outcells = []
  changecells = []
  hiddencells = []
  cellcells = []
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
  hasembed, hasshift = haskey(model, :embed), haskey(model, :dec_shifts)
  alpha, beta  = hasshift ? (model[:dec_shifts][1], model[:dec_shifts][2]) : (nothing, nothing)
  for i = 1:seqlen
    #dims data = [batchsize, V] = [batchsize, 5]
    input = hcat(prev_coords, z)
    if hasembed
      input = input * model[:embed]
    end
    state, gates = lstm(model[:decode], state, input; alpha=alpha, beta=beta, exposed=true)
    #push!(hiddencells,  Array(state[1]))
    #push!(cellcells,  Array(tanh(state[2])))
    output =  predict(model[:output], state[1]) #get output params
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm = get_mixparams(output, M; samplemode=true) #get mixture parameters and normalized logit values
    idx = get_pi_idx(rand(), pnorm; temp=temp, greedy=greedy)
    idx_eos = get_pi_idx(rand(), qnorm; temp=temp, greedy=greedy)
    eos = [0 0 0]
    eos[idx_eos] = 1
    if idx_eos != 3
      push!(hiddencells,  Array(state[1]))
      push!(cellcells,  Array(tanh(state[2])))
    end
    #=if idx_eos != 3
      push!(forgetcells,  Array(gates[1]))
      push!(incells,  Array(gates[2]))
      push!(outcells,  Array(gates[3]))
      push!(changecells,  Array(gates[4]))
    end =#
    next_x, next_y = sample_gaussian_2d(mu_x[idx], mu_y[idx], sigma_x[idx], sigma_y[idx], rho[idx]; temp = sqrt(temp), greedy=greedy)
    push!(mixture_params, [ pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm ])
    cur_coords = [next_x next_y eos[1] eos[2] eos[3]]
    points[:, i] = cur_coords'
    prev_coords = atype(copy(cur_coords))
  end
  return points, mixture_params, (hiddencells, cellcells)#(forgetcells, incells, outcells, changecells)
end

function decode(model, z; draw_mode = true, temperature = 1.0, factor = 0.2, greedy_mode = false)
  max_seq_length = 60
  sampled_points, mixture_params = sample(model, z; seqlen=max_seq_length,temperature=temperature, greedy_mode=greedy_mode)
  sampled_points = clean_points(sampled_points)
  sketch = constructsketch(sampled_points)
  if draw_mode
    savesketch(sketch, "sampled.png")
  end
end


function strokes_as_sketch_objs(sketch)
  label = "sampled stroke"
  recognized = false
  key_id = "sampled stroke"
  sketch_objs = []
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    points  = sketch.points[:, start_ind:end_ind]
    stroke = Sketch(label, recognized, key_id, points,  [0 size(points, 2)])
    push!(sketch_objs, stroke)
  end
  return sketch_objs
end

function classifystrokes(sketch)
  label = "sampled sketch"
  recognized = false
  key_id = "sampled sketch"
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    points  = sketch.points[:, start_ind:end_ind]
    idm = extractidm(points, [0 size(points, 2)])
    ypred = SVR.predict(svmmodel, idm')
    println(ypred)
    stroke = Sketch(label, recognized, key_id, points,  [0 size(points, 2)])
    savesketch(stroke, "stroke$(strokenum)-$(labels[Int(ypred[1])]).png")
  end
end

function stroke_decode(model, z_vecs, lens; draw_mode = true, temperature = 1.0, factor = 0.2, greedy_mode = false, strokes = nothing)
  max_seq_length = 60
#=  gatenames = ("forget", "ingate", "outgate", "change")
  mins = (0.1, 0.1, 0.1, -0.8)
  maxs = (0.9, 0.9, 0.9, 0.8)=#
  gatenames = ("hidden", "cell")
  maxs = (0.9, 0.9)
  mins = (-0.9, -0.9)
  sampled_points = []
  cells = []
  for i in 1:length(z_vecs)
    z = z_vecs[i]
    len = lens[i]
    sampled_stroke_points, mixture_params, cell = sample(model, z; seqlen=len + 5,temperature=temperature, greedy_mode=greedy_mode)
    sampled_stroke_points = stroke_clean_points(sampled_stroke_points)
    push!(cells, cell)
    #println(size(sampled_stroke_points))
    push!(sampled_points, sampled_stroke_points)
  end
  #sampled_points = stroke_clean_points(sampled_points)
  sketch = stroke_constructsketch(strokes)
  #sketch = stroke_constructsketch(sampled_points)
  strokes_objs = strokes_as_sketch_objs(sketch)
  for i = 1:length(strokes_objs)
    for j = 1:length(gatenames)
      save_saturated_inds(cells[i][j], gatenames[j], strokes_objs[i]; mincutoff = mins[j], maxcutoff = maxs[j])
    end
  end
  #classifystrokes(sketch)
  if draw_mode
    savesketch(sketch, "sampled.png")
  end
end

function getrandomsketch(points3D, idmtuples; tosave = true)
  idx = rand(1:length(points3D))
  #idx  = 2889
  #idx = 58
#  idx = 1339
  info("Selected index is $(idx)")
  x_5D = to_big_points(points3D[idx]; max_len = 50)
  sequence = makesequence(x_5D)
  if tosave
    i = 1
    for im in idmtuples[idx].stroke_ims
      saveidm(im,"strokeidms/idm$(idx)-$(i).png")
      i += 1
    end
    saveidm(idmtuples[idx].avg_im, "strokeidms/idm$(idx)-avg.png")
  end
  avidm, strokeidms = idm_indices_to_batch(idmtuples, idx, nothing)
  sequence = paddall([sequence], [(avidm, strokeidms)], 24)
  return map(a->convert(atype, a), sequence[1]), x_5D
end

function makesequence(points5D)
  sequence = []
  push!(sequence, [0 0 1 0 0])
  for i=1:size(points5D, 2)
    push!(sequence, points5D[:, i]')
  end
  return sequence
end

function tostrokesketch(points3D, idx)
  x_5D = to_big_points(points3D[idx]; max_len = 60)
  end_indices =  find(x_5D[4, :] .== 1)
  push!(end_indices, size(x_5D, 2))
  strokecount = Int(sum(x_5D[4, :]))
  stroke_start = 1
  strokeseq = []
  sseqlens = []
  for i = 1:strokecount
    tmp = makesequence(x_5D[:, stroke_start:end_indices[i]]) #IS THIS CORRECT? what about [0,0,0,1,0]?
    stroke_as_seq =  map(a->convert(atype, a), tmp)
    push!(strokeseq, stroke_as_seq)
    push!(sseqlens, length(stroke_start:end_indices[i]))
    stroke_start = end_indices[i] + 1
  end
  return strokeseq, sseqlens, x_5D
end

function rand_strokesketch(points3D)
  idx = rand(1:length(points3D))
  #idx  = 2889
  #idx = 58
  #idx = 2178
#  idx = 2388
  info("Selected index is $(idx)")
  return tostrokesketch(points3D, idx)
end

function get_strokelatentvecs(model, strokeseq)
  z_vecs = []
  for stroke in strokeseq
    push!(z_vecs, getlatentvector(model, stroke))
  end
  return z_vecs
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="Sketch sampler from model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--svmmodel"; arg_type=String; default="airplane.model"; help="Name of the pretrained svm model")
    ("--model"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--dataset"; arg_type=String; default="r_full_simplified_airplane.jld"; help="Name of the dataset")
    ("--T"; arg_type=Float64; default=1.0; help="Temperature.")
    ("--greedy"; action=:store_true; help="is data preprocessed and ready")
    ("--imlen"; arg_type=Int; default=0; help="Image dimentions.")
    ("--statmode"; action=:store_true; help="look at cells")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  w = load("$(pretrnp)$(o[:model])")
  model = revconvertmodel(w["model"])
  global svmmodel = SVR.loadmodel(o[:svmmodel])
  global labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  info("Model was loaded")
  trnpoints3D, vldpoints3D, tstpoints3D = loaddata("$(datap)data$(o[:imlen])$(o[:dataset])")
#  trnidm, vldidm, tstidm  = loaddata("$(datap)idm$(o[:imlen])$(o[:dataset])")
  info("Train, Valid, Test data obtained")
  x, lens, x_5D = rand_strokesketch(tstpoints3D)
  end_indices =  find(x_5D[4, :] .== 1)
  s = 1
  strokes = []
  for i= 1:length(end_indices)
    stroke = hcat(x_5D[:, s:end_indices[i] ], [0 0 0 0 1]')
    printpoints(stroke)
    push!(strokes, stroke)
    s = end_indices[i] + 1
  end

  if o[:statmode]
    println("In stat mode!")
    for i = 1:length(tstpoints3D)
      x, lens, x_5D = tostrokesketch(tstpoints3D, i)
      end_indices =  find(x_5D[4, :] .== 1)
      s = 1
      strokes = []
      for i= 1:length(end_indices)
        stroke = hcat(x_5D[:, s:end_indices[i] ], [0 0 0 0 1]')
        printpoints(stroke)
        push!(strokes, stroke)
        s = end_indices[i] + 1
      end
      z_vecs = get_strokelatentvecs(model, x)
      stroke_decode(model, z_vecs, lens; temperature=o[:T], greedy_mode=o[:greedy], strokes = strokes)
    end
    return
  end

  sketch = stroke_constructsketch(strokes)
  info("Random sketch was obtained")
  savesketch(sketch, "original.png")
  z_vecs = get_strokelatentvecs(model, x)
  info("got latent vector(s)")
  stroke_decode(model, z_vecs, lens; temperature=o[:T], greedy_mode=o[:greedy], strokes = strokes)
  SVR.freemodel(svmmodel)
end
main()
end
