include("DrawNet.jl")
include("Segmenter.jl")
include("../utils/Drawing.jl")

module Drawer
using DrawNet, Drawing, DataLoader
using Distributions, PyPlot, SVR
using Knet, ArgParse, JLD, JSON
using IDM, Segmenter
include("../rnns/RNN.jl")
include("Ferret.jl")
include("DataManager.jl")

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
    points2D = []
    x = 0
    y = 0
    for i=1:size(stroke, 2)
      x += Float64(stroke[1, i])
      y += Float64(stroke[2, i])
      if ( length(stroke[:, i]) == 4 && stroke[3, i] == 0 )  || ( length(stroke[:, i]) == 5 && stroke[5, i] == 1 ) #condition for V=4 -> stroke[3, i] == 0 for V=5 stroke[5, i] == 1
        push!(endidxs, endidxs[end] + (i-1))
        break
      end
      push!(points2D, x)
      push!(points2D, y)
    #  points2D[1, i] = x
    #  points2D[2, i] = y
    end
    points2D = reshape(points2D, (2, Int(length(points2D)/2)))
    if allpoints == nothing
      allpoints = points2D
    else
      allpoints = hcat(allpoints, points2D)
    end
    println(endidxs)
    println(size(allpoints))
  end
  label = "sampled sketch"
  recognized = false
  key_id = "sampled sketch"
  println(endidxs)
  #println(size(allpoints))
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
  V = size(model[:fw_embed], 1)
  M = Int((size(model[:output][1], 2)-(V-2))/6) #number of mixtures
  d_H = size(model[:output][1], 1) #decoder hidden unit size
  z_size = size(model[:z][1], 1) #size of latent vector z
  hiddencells = []
  cellcells = []
  if z == nothing
    z = randn(1, z_size)
  end
  points = zeros(V, seqlen)
  mixture_params = []
  prev_coords = zeros(1, V)
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
    output =  predict(model[:output], state[1]) #get output params
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm = get_mixparams(output, M, V; samplemode=true) #get mixture parameters and normalized logit values
    idx = get_pi_idx(rand(), pnorm; temp=temp, greedy=greedy)
    idx_eos = get_pi_idx(rand(), qnorm; temp=temp, greedy=greedy)
    eos = zeros(1, V-2)
    eos[idx_eos] = 1
    if idx_eos != 3
      push!(hiddencells,  Array(state[1]))
      push!(cellcells,  Array(tanh(state[2])))
    end
    next_x, next_y = sample_gaussian_2d(mu_x[idx], mu_y[idx], sigma_x[idx], sigma_y[idx], rho[idx]; temp = sqrt(temp), greedy=greedy)
    push!(mixture_params, [ pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm ])
    cur_coords = [next_x next_y eos]
    #println(cur_coords)
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
    sampled_stroke_points, mixture_params, cell = sample(model, z; seqlen=len+5, temperature=temperature, greedy_mode=greedy_mode)
    #println(size(sampled_stroke_points))
    #printpoints(sampled_stroke_points)
    sampled_stroke_points = clean_points(sampled_stroke_points)
    #println("after clean up")
    #printpoints(sampled_stroke_points)
    push!(cells, cell)
    #println(size(sampled_stroke_points))
    push!(sampled_points, sampled_stroke_points)
  end
  #sampled_points = stroke_clean_points(sampled_points)
  #sketch = stroke_constructsketch(strokes)
  sampledsketch = stroke_constructsketch(sampled_points)
  #classifystrokes(sketch)
  if draw_mode
    savesketch(sampledsketch, "sampled.png")
  end
  return sampledsketch
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
  #=Generate a neural network input sequence=#
  sequence = []
  push!(sequence, [0 0 1 0 0])
  for i=1:size(points5D, 2)
    push!(sequence, points5D[:, i]')
  end
  push!(sequence, [0 0 0 0 1])
  return sequence
end


function tostrokesketch(points3D, idx)
  #=Takes 3D points as an input and generates strokes as a sequences for neural network=#
  x_5D = to_big_points(points3D[idx]; max_len = 150)
  end_indices =  find(x_5D[4, :] .== 1)
  push!(end_indices, size(x_5D, 2))
  strokecount = Int(sum(x_5D[4, :]))
  stroke_start = 1
  strokeseq, sseqlens = [], []
  for i = 1:strokecount
    tmp = makesequence(x_5D[:, stroke_start:end_indices[i]]) #IS THIS CORRECT? what about [0,0,0,1,0]?
    stroke_as_seq =  map(a->convert(atype, a), tmp)
    push!(strokeseq, stroke_as_seq)
    push!(sseqlens, length(stroke_as_seq))
    stroke_start = end_indices[i] + 1
  end
  return strokeseq, sseqlens, x_5D
end

function rand_strokesketch(points3D)
  idx = rand(1:length(points3D))
  info("Selected index is $(idx)")
  return tostrokesketch(points3D, idx)
end

function get_strokelatentvecs(model, strokeseq)
  #=Retrieves latent vectors for each stroke in a sequence (strokeseq)=#
  z_vecs = []
  for stroke in strokeseq
    push!(z_vecs, getlatentvector(model, stroke))
  end
  return z_vecs
end

function randstrokes(sketches, scalefactor)
  idx = rand(1:length(sketches))
  #idx = 3127
  println("Index of the sketch $(idx)")
  sketch = sketches[idx]
  println(sketch.points')
  for ind in (sketch.end_indices[1:end-1]+1)
    println(sketch.points[:, ind]')
  end

  strokes = get_3d_strokes(sketch)
   #normalize points
   for stroke in strokes
     printpoints(stroke)
     stroke[1:2, :] /= scalefactor
     printpoints(stroke)
   end
   println("original points were printed")
   return strokes
end


function rand4d_strokesketch(points3D)
  idx = rand(1:length(points3D))
  #idx  = 2889
  #idx = 58
  #idx = 2178
#  idx = 2388
#  idx = 4389
#  idx =  4284
#idx = 933
#  idx = 4946
#  idx = 8244
  info("Selected index is $(idx)")
  return points3D2seq4d(points3D, idx)
end

function makesequence4d(points5D)
  sequence = []
  push!(sequence, [0 0 1 0])
  for i=1:size(points5D, 2)
    push!(sequence, [points5D[1:2, i]' 1 0])
    println("points",[points5D[1:2, i]' 1 0])
  end
  push!(sequence, [0 0 0 1])
  return sequence
end

function points3D2seq4d(points3D, idx)
  x_5D = to_big_points(points3D[idx]; max_len = 150)
  end_indices =  find(x_5D[4, :] .== 1)
  push!(end_indices, size(x_5D, 2))
  strokecount = Int(sum(x_5D[4, :]))
  stroke_start = 1
  strokeseq = []
  sseqlens = []
  for i = 1:strokecount
    tmp = makesequence4d(x_5D[:, stroke_start:end_indices[i]]) #IS THIS CORRECT? what about [0,0,0,1,0]?
    stroke_as_seq =  map(a->convert(atype, a), tmp)
    push!(strokeseq, stroke_as_seq)
    push!(sseqlens, length(strokeseq))
    stroke_start = end_indices[i] + 1
  end
  return strokeseq, sseqlens, x_5D
end

function getindices()
  d = Dict()
  d["firetruck"] = [8 29 950 1081 1450 1503]
  d["flower"]    = [364 416 899 941 1000]
  d["airplane"] = [20 103 640 775 915 978]
  d["cat"] = [1483 184 1005 1219 1574 1517]
  d["chair"] = [854 995 1003 2441]
  d["pig"] = [28 48 72 74]
  d["owl"] = [36 85 95 107 147]
  d["face"] = [52 373 767 876 893 1001]
  return d
end

function getcolor(T)
  if T <= 0.2
    return "#4286f4"
  elseif T <= 0.4
    return "#5738e2"
  elseif T <= 0.6
    return "#9338e2"
  elseif T <= 0.8
    return  "#da38e2"
  elseif T < 1
    return "#bc0353"
  end
  return "#e50909"
end

function generate_all_sketces(o)
  d = getindices()
  for key in keys(d)
    indices = d[key]
    dataset = loaddataset("$(datap)datasetr_full_simplified_$(key).jld")
    trnpoints3D, vldpoints3D, tstpoints3D = dataset[:trn][1], dataset[:vld][1], dataset[:tst][1]
    w = nothing
    #=for i = 15:-5:10
      if isfile("$(pretrnp)m$(i)$(key)_strokernnV5.jld")
        w = load("$(pretrnp)m$(i)$(key)_strokernnV5.jld")
        break
      end
    end=#
    w = load("$(pretrnp)m10$(key)_strokernnV5.jld")
    model = revconvertmodel(w["model"])
    for idx in indices
      x, lens, x_5D = tostrokesketch(tstpoints3D, idx)
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
      #sketch = stroke_constructsketch(strokes)
      #c = "black"
      #savesketch(sketch, "drawn/orig$(key)$(idx)$(Int(100*o[:T])).png"; color=c)
      sampledsketch = stroke_decode(model, z_vecs, lens; draw_mode = false, temperature=o[:T], greedy_mode=o[:greedy], strokes = strokes)
      c = getcolor(o[:T])
      savesketch(sampledsketch, "drawn/$(key)$(idx)$(Int(100*o[:T])).png"; color=c)
    end
  end
end


function getbestindices()
  d = Dict()
  d["firetruck"] = [29]
  d["flower"]    = [1000]
  d["airplane"] = [103]
  d["cat"] = [1005]
  d["chair"] = [854]
  d["pig"] = [48]
  d["owl"] = [ 147]
  d["face"] = [1001]
  return d
end

function generate_single(o)
  d = getbestindices()
  rawname = split(o[:a_filename], ".")[1]
  w = load("$(pretrnp)m10$(rawname)_strokernnV5.jld")
  model = revconvertmodel(w["model"])
  for key in keys(d)
    indices = d[key]
    dataset = loaddataset("$(datap)datasetr_full_simplified_$(key).jld")
    trnpoints3D, vldpoints3D, tstpoints3D = dataset[:trn][1], dataset[:vld][1], dataset[:tst][1]
    for idx in indices
      x, lens, x_5D = tostrokesketch(tstpoints3D, idx)
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
      sampledsketch = stroke_decode(model, z_vecs, lens; draw_mode = false, temperature=o[:T], greedy_mode=o[:greedy], strokes = strokes)
      c = "#1b9b3b"
      savesketch(sampledsketch, "inter/$(rawname)$(key)$(idx).png"; color=c)
    end
  end
end

function dict2sketch(dictdata)
  sketches = []
  for d in dictdata
    sketch = getsketch(d)
    push!(sketches, sketch)
  end
  return sketches
end

function get_seg_3d(o, labels; tt = nothing, params = nothing, dpath = annotp, scalefactor = 45)
  filename = string(dpath, o[:a_filename])
  #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  #annotations, annot_dicts = getannotateddata(filename, labels)
  sketches = get_sketch_objects(filename)
  acount = length(sketches)
  tt = (tt == nothing) ? randperm(acount) : tt
  #get training set size for train-test split
  trnsize = acount - div(acount, o[:cvfolds])
  println("Number of annotated data: $(acount) Training set size: $(trnsize)")
  trnsketches, tstsketches = data_tt_split(sketches, trnsize; rp = tt)

  #trn_dicts, tst_dicts = data_tt_split(annot_dicts, trnsize; rp = tt)
  #trnsketches, tstsketches = dict2sketch(trn_dicts), dict2sketch(tst_dicts)
  trnpoints3D, _, _ = preprocess(trnsketches, params)
  tstpoints3D, _, tstsketches = preprocess(tstsketches, params)
  for i = 1:length(tstsketches)
    tmpsk = Sketch(tstsketches[i].label, tstsketches[i].recognized, tstsketches[i].key_id, tstsketches[i].points, tstsketches[i].end_indices)
    savesketch(tmpsk, "segByModel/a_f$(o[:fold])_$(i).png")
  end
  DataLoader.normalize!(trnpoints3D, params; scalefactor = scalefactor)
  DataLoader.normalize!(tstpoints3D, params; scalefactor = scalefactor)
  return trnpoints3D, tstpoints3D
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="Sketch sampler from model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--svmmodel"; arg_type=String; default="airplane.model"; help="Name of the pretrained svm model")
    ("--model"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--gmodel"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--dataset"; arg_type=String; default="r_full_simplified_airplane.jld"; help="Name of the dataset")
    ("--T"; arg_type=Float64; default=1.0; help="Temperature.")
    ("--V"; arg_type=Int; default=5; help="vocab size.")
    ("--greedy"; action=:store_true; help="is data preprocessed and ready")
    ("--imlen"; arg_type=Int; default=0; help="Image dimentions.")
    ("--statmode"; action=:store_true; help="look at cells.")
    ("--segmentmode"; action=:store_true; help="segmentation mode.")
    ("--filename"; arg_type=String; default="airplane.ndjson"; help="Data file name.")
    ("--a_filename"; arg_type=String; default="airplane1014.ndjson"; help="Annotated data file name.")
    ("--batchsize"; arg_type=Int; default=1; help="Minibatch size.")
    ("--hascontext"; action=:store_true; help="Store true if context info is used")
    ("--cvfolds"; arg_type=Int; default=5; help="Number of folds to use for cross validation.")
    ("--fold"; arg_type=Int; default=-1; help="Current fold.")
    ("--a_datasize"; arg_type=Int; default=0; help="Annotated dataset size to use.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  if o[:segmentmode]
    if o[:hascontext]
      #load model for genrating sketches
      println("Loading generative model from $(pretrnp)$(o[:gmodel])")
      w = load("$(pretrnp)$(o[:gmodel])")
      genmodel = revconvertmodel(w["model"])
    end
    println("Loading segmentation model from $(pretrnp)$(o[:model])")
    w = load("$(pretrnp)$(o[:model])")
    model = revconvertmodel(w["model"])
    println("In segmentation mode")
    vldsize = 1 / o[:cvfolds]
    #scalefactor = 48.290142 #firetruck
    #scalefactor = 56.090145 #chair
    #scalefactor = 31.883362 #flower
    #scalefactor = 43.812866 #airplane
    scalefactor = 49.193924 #cat
    params = Parameters()
    params.batchsize = o[:batchsize]
    params.min_seq_length = 1
    params.max_seq_length = 200
    filename = string(annotp, o[:a_filename])
    rawname = split(o[:a_filename], ".")[1]
    tt = load("annotsplits/$(rawname)indices.jld")["indices"]
    println("Fold $(o[:fold])")
    for i=1:(o[:fold]-1)
      println("Shifting $(i)")
      tt = getshiftedindx(tt, o)
    end
    dpath = annotp
    if dpath == huangp
      categories = getHuangLabels()
      labels = categories[rawname]
      println(labels)
    elseif dpath == annotp
      categories = getGoogleLabels()
      labels = categories[rawname]
      println(labels)
    end
    classnames = getGoogleSegmentNames()
    classnames = copy(classnames[rawname])
    colors = getGoogleColors()
    colors = colors[rawname]
    o[:numclasses] = length(labels)
    #what is the scale factor?
    trnpoints3D, tstpoints3D = get_seg_3d(o, labels; tt = tt, params=params, dpath = dpath, scalefactor=scalefactor)
    println("Data retrieval checkpoint")
    sketchpoints3D = tstpoints3D
    #return
    #DataLoader.normalize!(sketchpoints3D, params; scalefactor=scalefactor)
    info("Number of sketches = $(length(sketchpoints3D))")
    for i = 1:length(sketchpoints3D)
      x, lens, x_5D = tostrokesketch(sketchpoints3D, i)
      end_indices =  find(x_5D[4, :] .== 1)
      s = 1
      strokes = []
      for j = 1:length(end_indices)
        stroke = hcat(x_5D[:, s:end_indices[j] ], [0 0 0 0 1]')
        #printpoints(stroke)
        push!(strokes, stroke)
        s = end_indices[j] + 1
      end
      sketch = stroke_constructsketch(strokes)
      savesketch(sketch, "segByModel/o$(rawname)_f$(o[:fold])_$(i).png")
      strokeclasses = getstrokelabels(model, x, lens; genmodel=genmodel)
      println(strokeclasses)
      @assert(length(strokes) == length(strokeclasses))
      saveslabeled(sketch, strokeclasses, classnames, colors, "segByModel/$(rawname)_f$(o[:fold])_$(i).png")
    end
    println("Segmentation fold finish checkpoint")
    return
    x, lens, x_5D = rand_strokesketch(sketchpoints3D)
    end_indices =  find(x_5D[4, :] .== 1)
    s = 1
    strokes = []
    for i= 1:length(end_indices)
      stroke = hcat(x_5D[:, s:end_indices[i] ], [0 0 0 0 1]')
      printpoints(stroke)
      push!(strokes, stroke)
      s = end_indices[i] + 1
    end
    sketch = stroke_constructsketch(strokes)
    info("Random sketch was obtained")
    strokeclasses = getstrokelabels(model, x, lens)
    println(strokeclasses)
    saveslabeled(sketch, strokeclasses, classnames, "original.png")
    return
  end
  
  #generate_all_sketces(o)
  #generate_single(o)
  #return
  println("Loading model from $(pretrnp)$(o[:model])")
  w = load("$(pretrnp)$(o[:model])")
  model = revconvertmodel(w["model"])
  global svmmodel = SVR.loadmodel(o[:svmmodel])
  #global labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  #global labels = [  "W", "B", "T" ,"WNDW", "FA"]
  #classnames = ["wing", "body", "tail", "window", "full airplane"]
  #global labels = [ "LGT", "LDR", "B", "C", "WNDW", "WHS",  "WHL"] #labels for firetruck
 # classnames = [ "LGT", "LDR", "B", "C", "WNDW", "WHS",  "WHL"]
  global labels = [ "EAR", "H", "EYE", "N", "W", "M",  "B", "T", "L"]
  classnames = [ "EAR", "H", "EYE", "N", "W", "M",  "B", "T", "L"]
  #global labels = [ "L", "F", "FP"]
  #classnames = ["leaf", "fruit", "full pineapple"]
  info("Model was loaded")
  dataset = loaddataset("$(datap)dataset$(o[:dataset])")
  trnpoints3D, vldpoints3D, tstpoints3D = dataset[:trn][1], dataset[:vld][1], dataset[:tst][1]
  rawname = split(o[:a_filename], ".")[1]

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
      for j= 1:length(end_indices)
        stroke = hcat(x_5D[:, s:end_indices[j] ], [0 0 0 0 1]')
        push!(strokes, stroke)
        s = end_indices[j] + 1
      end
      #z_vecs = get_strokelatentvecs(model, x)
      sketch = stroke_constructsketch(strokes)
      savesketch(sketch, "pics/o$(rawname)$(i).png")
      #stroke_decode(model, z_vecs, lens; temperature=o[:T], greedy_mode=o[:greedy], strokes = strokes)
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


if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "Drawer.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
end
