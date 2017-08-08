include("../utils/DataLoader.jl")
include("../idm/IDM.jl")
module DrawNet
using Drawing, DataLoader, IDM
using Knet, ArgParse, JLD, AutoGrad
include("../rnns/RNN.jl")
include("../models/FullStrokeRNN.jl")

type KLparameters
  w::AbstractFloat
  wstart::AbstractFloat
  decayrate::AbstractFloat
end

type LRparameters
  lr::AbstractFloat
  minlr::AbstractFloat
  decayrate::AbstractFloat
end

#=
Bivariate normal distribution pdf.
=#
function bivariate_prob(delta_x, delta_y, mu_x, mu_y, sigma_x, sigma_y, rho)
  z = ((delta_x - mu_x)/sigma_x)^2 + ((delta_y - mu_y)/sigma_y)^2 - 2*rho*((delta_y - mu_y)/sigma_y)*((delta_x - mu_x)/sigma_x)
  t = sqrt(1-rho^2)
  prob = exp( -( z/(2*t*t) ) ) / (2*pi*sigma_x*sigma_y*t)
  return prob
end

#=
Vectorized bivariate normal distribution pdf.
=#
function vec_bivariate_prob(x1, x2, mu1, mu2, s1, s2, rho)
  norm1 = x1 .- mu1
  norm2 = x2 .- mu2
  s1s2 = s1 .* s2
  z = (norm1./s1).*(norm1./s1) + (norm2./s2).*(norm2./s2) - ( (2 .* rho .*  (norm1 .* norm2)) ./ s1s2 )
  neg_rho = 1 .- rho.*rho
  prob = exp(-(z ./ (2.*neg_rho)) ) ./ (2*pi.*s1s2.*sqrt(neg_rho))
  return prob
end

function softmax(p, d::Int)
  tmp = exp(p)
  return tmp ./ sum(tmp, d)
end

function initstate(batchsize, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), batchsize, length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), batchsize, length(c)), 0)
    return (h,c)
end

function encode(model, data, maxlen::Int, batchsize::Int; dprob = 0)
  #Initialize states for forward-backward rnns
  statefw = initstate(batchsize, model[:fw_state0])
  statebw = initstate(batchsize, model[:bw_state0])
  #forward encoder
  w = AutoGrad.getval(model)
  hasembed, hasshift = haskey(w, :fw_embed), haskey(w, :fw_shifts)
  alpha, beta  = hasshift ? (model[:fw_shifts][1], model[:fw_shifts][2]) : (nothing, nothing)
  for i = 1:maxlen
    input = hasembed ? data[i] * model[:fw_embed] : data[i]
    statefw = lstm(model[:fw_encode], statefw, input; alpha=alpha, beta=beta, dprob=dprob)
  end
  #backward encoder
  hasembed, hasshift = haskey(w, :bw_embed), haskey(w, :bw_shifts)
  alpha, beta  = hasshift ? (model[:bw_shifts][1], model[:bw_shifts][2]) : (nothing, nothing)
  for i = maxlen:-1:1
    input = hasembed ? data[i]*model[:bw_embed] : data[i]
    statebw = lstm(model[:bw_encode], statebw, input; alpha=alpha, beta=beta, dprob=dprob)
  end
  return hcat(statefw[1], statebw[1]) #(h_fw, c_fw) = statefw, (h_bw, c_bw) = statebw
end

function get_mixparams(output, M::Int; samplemode=false)
  #Here I used different ordering for outputs; in practice order doesn't matter
  pnorm = softmax(output[:, 1:M], 2) #normalized distribution probabilities
  mu_x = output[:, M+1:2M]
  mu_y = output[:, 2M+1:3M]
  sigma_x = exp(output[:, 3M+1:4M])
  sigma_y = exp(output[:, 4M+1:5M])
  rho = tanh(output[:, 5M+1:6M])
  if samplemode
    qnorm = softmax(output[:, 6M+1:6M+3], 2) #normalized log probabilities of logits
  else
    qnorm = logp(output[:, 6M+1:6M+3], 2) #normalized log probabilities of logits
  end
  return pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm
end
#=
inputpoints - list of (1, 5) point tuples
model - pretrained model
=#
function getlatentvector(model, inputpoints)
  #model settings
  seqlen = length(inputpoints)
  z_size = size(model[:z][1], 1) #size of latent vector z
  h = encode(model, inputpoints, seqlen, 1)
  #compute latent vector
  mu = h*model[:mu][1] .+ model[:mu][2]
  sigma_cap = h*model[:sigma_cap][1] .+ model[:sigma_cap][2]
  sigma = exp( sigma_cap/2 )
  z = mu + sigma .* atype( gaussian(1, z_size; mean=0, std=1) )
  return z
end

function predict(param, input)
  return input * param[1] .+ param[2]
end

function paddall(data, idmtuples, imlen)
  println("padding idms")
  newdata = []
  for i = 1:length(data)
    x = data[i]
    (avgidm, strokeidms) = idmtuples[i]
    newx = padidms(x, strokeidms, imlen)
    push!(newdata, newx)
  end
  @assert(length(data) == length(newdata))
  return newdata
end

function padidms(x, strokeidms, imlen)
  (batchsize, V) = size(x[1])
  idmsize = imlen^2
  newx = []
  for i = 1:length(x)
    push!(newx, zeros(batchsize, V+idmsize))
  end
  for i = 1:batchsize
    curstrokes = strokeidms[i]
    k = 1
    for j = 1:length(x)
      newx[j][i,1:V] = x[j][i, :]
      if x[j][i, 5] == 1
        continue
      end
      newx[j][i,V+1:V+idmsize] = curstrokes[k]
      if x[j][i, 3] == 0
        k += 1
      end
    end
  end
  return newx
end

#random scaling of x and y values
function perturb(data; scalefactor=0.1)
  hasidm = size(data[1], 2) > 5
  pdata = []
  for i=1:length(data)
    x_scalefactor = (rand() - 0.5) * 2 * scalefactor + 1.0
    y_scalefactor = (rand() - 0.5) * 2 * scalefactor + 1.0
    result = copy(data[i])
    result[:, 1] *= x_scalefactor
    result[:, 2] *= y_scalefactor
    #perturb idm if needed
    if hasidm
      idm_scalefactor = (rand() - 0.5) * 2 * scalefactor + 1.0
      result[:, 6:size(data[i], 2)] *= scalefactor
    end
    push!(pdata, result)
  end
  return pdata
end


function getdata(filename, params)
  return getsketchpoints3D(filename; params = params)
end

function getstrokeseqs(x_batch_5D)
  max_stroke_count = 1
  (V, maxlen, batchsize)  = size(x_batch_5D)
  strokecount = zeros(1, batchsize)
  end_indices = []
  for i=1:batchsize
    push!(end_indices, find(x_batch_5D[4, :, i] .== 1) )
    strokecount[i] = Int( sum(x_batch_5D[4, :, i]) )
    max_stroke_count = max(max_stroke_count, strokecount[i])
  end
  max_stroke_count = Int(max_stroke_count)
  stroke_start = ones(Int, 1, batchsize)
  stroke_batch = []
  seqlens = []
  for i=1:max_stroke_count
    #find an i'th stroke with maximum length
    max_stroke_len = 1
    for j = 1:batchsize
      if i > length(end_indices[j])
        continue
      end
      max_stroke_len = max(max_stroke_len, length(stroke_start[j] : end_indices[j][i]) )
    end
    apstroke = [zeros(batchsize, V)]
    for j = 1:batchsize
      #if there are no strokes left
      if i > length(end_indices[j])
        apstroke[1][j, 5] = 1
      else
        apstroke[1][j, 3] = 1
      end
    end
    #predefine i'th strokes points
    stroke = []
    for k = 1:max_stroke_len
      push!(stroke, zeros(batchsize, V))
    end

    #for each batch element initialize its corresponding stroke
    for j = 1:batchsize
      x5D = x_batch_5D[:, :, j]' # This is current sketch points dims = [maxlen, V]

      #if there are no strokes left for current sketch
      if i > length(end_indices[j])
        for k = 1:length(stroke)
          stroke[k][j, :] = [0 0 0 0 1]
        end
        continue
      end
      s = stroke_start[j] #starting index i'th stroke of j'th sketch
      e = end_indices[j][i] #ending index of i'th stroke of j'th sketch
      #init points
      for k = s:e
      #  println("$(size(stroke[k-stroke_start[j]+1][j, :]))  $(size(x5D[k, :]))")
        stroke[k-s+1][j, :] = x5D[k, :]
      end
      #pad ends if needed
      for k = (e-s+2):length(stroke)
        stroke[k][j, :] = [0 0 0 0 1]
      end
      #DONOT FORGET TO MODIFY START[j]
      stroke_start[j] = end_indices[j][i] + 1
    end
    #=next_aps_stroke = zeros(batchsize, V)
    for pts in stroke
      next_aps_stroke[:, 1:2] += pts[:, 1:2] #ADD STRATING POINT? FIRST GO TO DRAWING TO SOLVE THE PROBLEM
    end=#

    #account fo the initial values for first stroke [0,0,0,1,0] already added?
    if i > 1
      append!(apstroke, stroke)
      push!(seqlens, max_stroke_len+1)
    else
      apstroke = stroke
      push!(seqlens, max_stroke_len)
    end
    #push!(seqlens, max_stroke_len+1)
    push!(stroke_batch, apstroke)
  end
  return stroke_batch, seqlens
end



#update learning rate of parameters
function updatelr!(opts::Associative, cur_lr)
  for (key, val) in opts
    if typeof(val) == Knet.Adam
      val.lr = cur_lr
    else
      for opt in val
        opt.lr = cur_lr
      end
    end
  end
end

function normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params::Parameters)
  DataLoader.normalize!(trnpoints3D, params)
  DataLoader.normalize!(vldpoints3D, params; scalefactor=params.scalefactor)
  DataLoader.normalize!(tstpoints3D, params; scalefactor=params.scalefactor)
end

function normalizeidms!(trnidm, vldidm, tstidm)
  s_avg, s_stroke = get_im_stds(trnidm)
  normalizeall!(trnidm, s_avg, s_stroke)
  normalizeall!(vldidm, s_avg, s_stroke)
  normalizeall!(tstidm, s_avg, s_stroke)
end

function normalizeall!(idmobjs, s_avg, s_stroke)
  for idm in idmobjs
    for strokeim in idm.stroke_ims
      strokeim /= s_stroke
    end
    idm.avg_im /= s_avg
  end
end

function savedata(filename, trndata, vlddata, tstdata)
  rawname = split(filename, ".")[1]
  save("../data/$(rawname).jld","train", trndata, "valid", vlddata, "test", tstdata)
end

function loaddata(filename)
  dataset = load(filename)
  println(filename)
  return dataset["train"], dataset["valid"], dataset["test"]
end

function get_splitteddata(data, trnidx, vldidx, tstidx)
  return data[trnidx], data[vldidx], data[tstidx]
end

# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T}, optim_type) = eval(parse(optim_type))
initoptim{T<:Number}(::Array{T}, optim_type) = eval(parse(optim_type))
initoptim(model::Associative, optim_type) = Dict( key => initoptim(val, optim_type) for (key,val) in model )
initoptim(w, optim_type) = map(x->initoptim(x, optim_type), w)


# convert model to save
convertmodel{T<:Number}(x::KnetArray{T}) = convert(Array{T}, x)
convertmodel{T<:Number}(x::Array{T}) = convert(Array{T}, x)
convertmodel(a::Associative)=Dict(k=>convertmodel(v) for (k,v) in a)
convertmodel(a) = map(x->convertmodel(x), a)

# reverse model converted for loading from file
revconvertmodel{T<:Number}(x::Array{T}) = convert(KnetArray{T}, x)
revconvertmodel(a::Associative) = Dict(k=>revconvertmodel(v) for (k, v) in a)
revconvertmodel(a) = map(x->revconvertmodel(x), a)

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="A Neural Representation of Sketch Drawings. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--epochs"; arg_type=Int; default=100; help="Total number of training set. Keep large.")
    ("--save_every"; arg_type=Int; default=10; help="Number of epochs per checkpoint creation.")
    ("--dec_model"; arg_type=String; default="lstm"; help="Decoder: lstm, or ....")
    ("--filename"; arg_type=String; default="full_simplified_airplane.ndjson"; help="Data file name")
    ("--bestmodel"; arg_type=String; default="bestmodel.jld"; help="File with the best model")
    ("--tmpmodel"; arg_type=String; default="tmpmodel.jld"; help="File with intermediate models")
    ("--dec_rnn_size"; arg_type=Int; default=2048; help="Size of decoder.")
    ("--enc_model"; arg_type=String; default="lstm"; help="Ecoder: lstm, or ....")
    ("--enc_rnn_size"; arg_type=Int; default=512; help="Size of encoder.")
    ("--batchsize"; arg_type=Int; default=100; help="Minibatch size. Recommend leaving at 100.")
    ("--grad_clip"; arg_type=Float64; default=1.0; help="Gradient clipping. Recommend leaving at 1.0.")
    ("--scalefactor"; arg_type=Float64; default=0.1; help="Random scaling data augmention proportion.")
    ("--num_mixture"; arg_type=Int; default=20; help="Number of mixtures in Gaussian mixture model.")
    ("--imlen"; arg_type=Int; default=0; help="Image dimentions.")
    ("--z_size"; arg_type=Int; default=128; help="Size of latent vector z. Recommend 32, 64 or 128.")
    ("--dprob"; arg_type=Float64; default=0.1; help="Dropout probability(keep prob = 1 - dropoutprob).")
    ("--V"; arg_type=Int; default=5; help="Number of elements in point vector.")
    ("--wkl"; arg_type=Float64; default=1.0; help="Parameter weight for Kullback-Leibler loss.")
    ("--kl_tolerance"; arg_type=Float64; default=0.05; help="Level of KL loss at which to stop optimizing for KL.") #KL_min
    ("--kl_decay_rate"; arg_type=Float64; default=0.99995; help="KL annealing decay rate per minibatch.") #PER MINIBATCH = R
    ("--kl_weight_start"; arg_type=Float64; default=0.01; help="KL start weight when annealing.")# n_min
    ("--lr"; arg_type=Float64; default=0.0001; help="Learning rate")
    ("--minlr"; arg_type=Float64; default=0.00001; help="Minimum learning rate.")
    ("--lr_decay_rate"; arg_type=Float64; default=0.99999; help="Minimum learning rate.")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
    ("--optimization"; default="Adam(;gclip = 1.0)"; help="Optimization algorithm and parameters.")
    ("--dataset"; arg_type=String; default="full_simplified_airplane.jld"; help="Name of the dataset")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  model = initmodel(o)
  params = Parameters()
  global optim = initoptim(model, o[:optimization])
  smooth = true
  if !o[:readydata]
    sketchpoints3D, numbatches, sketches = getdata(o[:filename], params)
    trnidx, vldidx, tstidx = splitdata(sketchpoints3D)
    info("data was split")
    trnpoints3D, vldpoints3D, tstpoints3D = get_splitteddata(sketchpoints3D, trnidx, vldidx, tstidx)
    trnsketches, vldsketches, tstsketches = get_splitteddata(sketches, trnidx, vldidx, tstidx)
    info("getting idm objects")
    trnidm = get_idm_objects(trnsketches; imlen = o[:imlen], smooth = smooth)
    vldidm = get_idm_objects(vldsketches; imlen = o[:imlen], smooth = smooth)
    tstidm = get_idm_objects(tstsketches; imlen = o[:imlen], smooth = smooth)
    info("In nomralization phase")
    normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params)
    #normalizeidms!(trnidm, vldidm, tstidm)
    savedata("idx$(o[:imlen])$(o[:filename])", trnidx, vldidx, tstidx)
    savedata("data$(o[:imlen])$(o[:filename])", trnpoints3D, vldpoints3D, tstpoints3D)
    savedata("idm$(o[:imlen])$(o[:filename])", trnidm, vldidm, tstidm)
    #save_idmtuples(o[:filename], trnpoints3D, vldpoints3D, tstpoints3D)
  else
    println("Loading data for training!")
    trnpoints3D, vldpoints3D, tstpoints3D = loaddata("$(datap)data$(o[:imlen])$(o[:dataset])")
  #  trnidm, vldidm, tstidm  = loaddata("$(datap)idm$(o[:imlen])$(o[:dataset])")
  end
  trnidm, vldidm, tstidm = nothing, nothing, nothing
  trn_batch_count = div(length(trnpoints3D), params.batchsize)
  params.numbatches = trn_batch_count
  trndata, trnseqlens = minibatch(trnpoints3D, trn_batch_count-1, params)
  vld_batch_count = div(length(vldpoints3D), params.batchsize)
  params.numbatches = vld_batch_count
  vlddata, vldseqlens= minibatch(vldpoints3D, vld_batch_count-1, params)

  tst_batch_count = div(length(tstpoints3D), params.batchsize)
  println("Starting training")
  reportmodel(model)
#  trndata = paddall(trndata, trnidmtuples, o[:imlen])
  #vlddata = paddall(vlddata, vldidmtuples, o[:imlen])
  #info("padding was complete")
  flush(STDOUT)
  train(model, trndata, trnseqlens, vlddata, vldseqlens, optim, o)
end
#main()
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "DrawNet.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

export revconvertmodel, encode, loaddata
export getlatentvector, predict, get_mixparams
export softmax, sample_gaussian_2d, paddall
end
