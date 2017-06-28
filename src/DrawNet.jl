include("../utils/DataLoader.jl")
module DrawNet
using Drawing, DataLoader
using Knet, ArgParse, JLD, AutoGrad

type KLparameters
  w::AbstractFloat
  weight_start::AbstractFloat
  decay_rate::AbstractFloat
  tolerance::AbstractFloat
end

type LRparameters
  lr::AbstractFloat
  min_lr::AbstractFloat
  decay_rate::AbstractFloat
end

global const atype = ( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} )
initxav(d...) = atype(xavier(d...))
initzeros(d...) = atype(zeros(d...))
initrandn(winit=0.0001, d...) = KnetArray{Float32}(winit*randn(d...))
#=
e_H -> size of hidden state of the encoder
d_H -> size of hidden state of the decoder
z_size -> size of latent vector z
V -> point vector size (i.e. 5 for (delta_x, delta_y, p1, p2, p3))
num_mixture -> number of gaussian mixtures
=#
function initmodel(e_H::Int, d_H::Int, V::Int, z_size::Int, num_mixture::Int)
  #initial hidden and cell states of forward encoder
  model = Dict{Symbol, Any}()
  info("Initializing encoder.")
  initencoder(model, e_H, V)
  info("Encoder was initialized. Initializing predecoder.")
  initpredecoder(model, e_H, d_H, z_size)
  info("Predecoder was initialized. Initializing decoder.")
  initdecoder(model, d_H, V, num_mixture, z_size)
  info("Initialization complete.")
  return model
end


#=
model -> rnn model
H -> size of hidden state of the encoder
V -> point vector size (i.e. 5 for (delta_x, delta_y, p1, p2, p3))
=#
function initencoder(model, H::Int, V::Int)
  #incoming input -> dims = (batchsize, V=5)
  model[:fw_state0] = [initxav(1, H), initzeros(1, H)]
  model[:fw_embed] = initxav(V, H) # x = input * model[:fw_embed]; x_dims = [batchsize, H]
  #here x and hidden will be concatenated form lstm_input with dims = [batchsize, H]
  model[:fw_encode] = [ initxav(2H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
  #same analysis goes for the decoder
  model[:bw_state0] = [initxav(1, H), initzeros(1, H)]
  model[:bw_embed] = initxav(V, H)
  model[:bw_encode] = [ initxav(2H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
end

#=
model -> rnn model
e_H -> size of hidden state of the encoder
d_H -> size of hidden state of the decoder
z_size -> size of latent vector z
=#
function initpredecoder(model, e_H::Int, d_H::Int, z_size::Int)
  #Incoming input dims = [batchsize, 2e_H]
  model[:mu] = [initxav(2e_H, z_size), initzeros(1, z_size)] #mu = input * W_mu .+ b_mu -> dims = [batchsize, z_size]
  model[:sigma_cap] = [initxav(2e_H, z_size), initzeros(1, z_size)] #sigma = input * W_sigma .+ b_sigma -> dims = [batchsize, z_size]
  #perform z = mu .+ sigma*N(0, I) -> z_dims = [batchsize, z_size]
  model[:z] = [initxav(z_size, 2d_H), initzeros(1, 2d_H)] # dec_H_0 = z*W_z .+ b_z -> dims = [batchsize, d_H]
end

#=
model -> rnn model
H -> size of hidden state of the decoder
V -> point vector size (i.e. 5 for (delta_x, delta_y, p1, p2, p3))
z_size -> size of latent vector z
num_mixture -> number of gaussian mixtures
=#
function initdecoder(model, H::Int, V::Int, num_mixture::Int, z_size::Int)
  initxav(d...) = atype(xavier(d...))
  #incoming input dims = [batchsize, z_size + V]
  model[:embed] = initxav(z_size + V, H) # x = input * model[:embed]; x_dims = [batchsize, H]
  model[:decode] = [ initxav(2H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
  model[:output] = [initxav(H, 6num_mixture + 3 ), initzeros(1, 6num_mixture + 3 )] #output = lstm_out * W_output .+ b_output -> dims = [batchsize, 6*num_mixture + 3]
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

function encode(model, data, maxlen::Int, batchsize::Int)
  #Initialize states for forward-backward rnns
  statefw = initstate(batchsize, model[:fw_state0])
  statebw = initstate(batchsize, model[:bw_state0])
  #forward encoder
  for i = 1:maxlen
    input = data[i] * model[:fw_embed]
    statefw = lstm(model[:fw_encode], statefw, input)
  end
  #backward encoder
  for i = maxlen:-1:1
    input = data[i]*model[:bw_embed]
    statebw = lstm(model[:bw_encode], statebw, input)
  end
  return hcat(statefw[1], statebw[1]) #(h_fw, c_fw) = statefw, (h_bw, c_bw) = statebw
end

function get_mixparams(output, M::Int)
  #Here I used different ordering for outputs; in practice order doesn't matter
  pnorm = softmax(output[:, 1:M], 2) #normalized distribution probabilities
  mu_x = output[:, M+1:2M]
  mu_y = output[:, 2M+1:3M]
  sigma_x = exp(output[:, 3M+1:4M])
  sigma_y = exp(output[:, 4M+1:5M])
  rho = tanh(output[:, 5M+1:6M])
  qnorm = logp(output[:, 6M+1:6M+3], 2) #normalized log probabilities of logits
  return pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm
end

#sequence to sequence variational autoencoder
function s2sVAE(model, data, seqlen, wkl, kl_tolerance; epsilon = 1e-6, istraining::Bool = true)
  #model settings
  maxlen = maximum(seqlen) #maximum length of the input sequence
  M = Int((size(model[:output][1], 2)-3)/6) #number of mixtures
  (batchsize, V) = size(data[1])
  d_H = size(model[:embed], 2) #decoder hidden unit size
  z_size = size(model[:z][1], 1) #size of latent vector z
  h = encode(model, data, maxlen, batchsize)
  #predecoder step
  mu = h*model[:mu][1] .+ model[:mu][2]
  sigma_cap = h*model[:sigma_cap][1] .+ model[:sigma_cap][2]
  sigma = exp( sigma_cap/2 )
  z = mu + sigma .* atype( gaussian(batchsize, z_size; mean=0, std=1) )

  #decoder step
  hc = tanh(z*model[:z][1] .+ model[:z][2])
  state = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
  penstate_loss = 0
  offset_loss = 0
  for i = 2:maxlen
    #dims data = [batchsize, V] = [batchsize, 5]
    input = hcat(data[i-1], z) #concatenate latent vector with previous point
    input = input * model[:embed]
    state = lstm(model[:decode], state, input)
    output =  predict(model[:output], state[1]) #get output params
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qnorm = get_mixparams(output, M) #get mixtur parameters and normalized logit values
    mix_probs = pnorm .* vec_bivariate_prob(data[i][:, 1], data[i][:, 2], mu_x, mu_y, sigma_x, sigma_y, rho) #weighted probabilities of mixtures
    mask = 1 .- data[i][:, V] #mask to zero out all terms beyond actual N_s the last actual stroke
    offset_loss += -sum( log( sum(mix_probs, 2).+ epsilon ) .* mask ) #L_s on paper(add epsilon to avoid log(0))
    if istraining
      penstate_loss += -sum(data[i][:, (V-2):V] .* qnorm) #L_p on paper
    else
      penstate_loss += -sum(data[i][:, (V-2):V] .* qnorm .* mask) #L_p on paper
    end
  end
  offset_loss /= (maxlen * batchsize)
  penstate_loss /= (maxlen * batchsize)
  kl_loss = -sum((1 + sigma_cap - mu.*mu - exp(sigma_cap))) / (2*z_size*batchsize)   #Kullback-Leibler divergence loss term
  if istraining
    kl_loss = max(kl_loss, kl_tolerance)
  else
    return penstate_loss, offset_loss, kl_loss
  end
  loss = offset_loss + penstate_loss + wkl*kl_loss
  return loss
end

function predict(param, input)
  return input * param[1] .+ param[2]
end


function initstate(batchsize, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), batchsize, length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), batchsize, length(c)), 0)
    return (h,c)
end

s2sVAEgrad = grad(s2sVAE)
function train(model, data, seqlens, opts, epochs, lrp::LRparameters, kl::KLparameters)
  cur_wkl, step = 0, 0
  for e = 1:epochs
    for i = 1:length(data)
      cur_wkl = kl.w - (kl.w - kl.weight_start) * ((kl.decay_rate)^step)
      cur_lr = (lrp.lr - lrp.min_lr)*(lrp.decay_rate^step) + lrp.min_lr
      grads = s2sVAEgrad(model, map(a->convert(atype, a), data[i]), seqlens[i], cur_wkl, kl.tolerance)
      updatelr!(opts, cur_lr)
      update!(model, grads, opts)
      step += 1
    end
    (rec_loss, kl_loss) = evaluatemodel(model, data, seqlens, kl.w, kl.tolerance)
    @printf("epoch: %d reconstuction loss: %g KL loss: %g  wkl: %g \n", e, rec_loss, kl_loss, cur_wkl)
    if e%10==0
      flush(STDOUT)
      arrmodel = convertmodel(model)
      save("../pretrained/model$(e).jld","model", arrmodel)
    end
  end

end

#calculates average reconstuction and KL losses
function evaluatemodel(model, data, seqlens, wkl, kl_tolerance)
  rec_loss, kl_loss = 0, 0
  for i = 1:length(data)
    penstate_loss, offset_loss, cur_kl_loss = s2sVAE(model, map(a->convert(atype, a), data[i]), seqlens[i], wkl, kl_tolerance; istraining = false)
    rec_loss += (penstate_loss + offset_loss)
    kl_loss += cur_kl_loss
  end
  return rec_loss/length(data), kl_loss/length(data)
end

function loaddata(filename, params)
  return getsketchpoints3D(filename; params = params)
end

function minibatch(sketchpoints3D, numbatches, params)
  info("Minibatching")
  data = []
  seqlens = []
  for i=0:(numbatches-1)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, params)
    sequence = []
    for j=1:size(x_batch_5D, 2)
      points = x_batch_5D[:, j, :]'
      push!(sequence, points)
    end
    push!(data, sequence)
    push!(seqlens, seqlen)
  end
  return data, seqlens
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

function savedata(filename, trndata, vlddata, tstdata)
  rawname = split(filename, ".")[1]
  save("../data/$(rawname).jld","train", trndata, "valid", vlddata, "test", tstdata)
end

function loaddata(filename)
  dataset = load(filename)
  return dataset["train"], dataset["valid"], dataset["test"]
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
    ("--save_every"; arg_type=Int; default=500; help="Number of batches per checkpoint creation.")
    ("--dec_model"; arg_type=String; default="lstm"; help="Decoder: lstm, or ....")
    ("--filename"; arg_type=String; default="full_simplified_airplane.ndjson"; help="Data file name")
    ("--dec_rnn_size"; arg_type=Int; default=2048; help="Size of decoder.")
    ("--enc_model"; arg_type=String; default="lstm"; help="Ecoder: lstm, or ....")
    ("--enc_rnn_size"; arg_type=Int; default=512; help="Size of encoder.")
    ("--batchsize"; arg_type=Int; default=100; help="Minibatch size. Recommend leaving at 100.")
    ("--grad_clip"; arg_type=Float64; default=1.0; help="Gradient clipping. Recommend leaving at 1.0.")
    ("--num_mixture"; arg_type=Int; default=20; help="Number of mixtures in Gaussian mixture model.")
    ("--z_size"; arg_type=Int; default=128; help="Size of latent vector z. Recommend 32, 64 or 128.")
    ("--V"; arg_type=Int; default=5; help="Number of elements in point vector.")
    ("--wkl"; arg_type=Float64; default=1.0; help="Parameter weight for Kullback-Leibler loss.")
    ("--kl_tolerance"; arg_type=Float64; default=0.2; help="Level of KL loss at which to stop optimizing for KL.") #KL_min
    ("--kl_decay_rate"; arg_type=Float64; default=0.99995; help="KL annealing decay rate per minibatch.") #PER MINIBATCH = R
    ("--kl_weight_start"; arg_type=Float64; default=0.01; help="KL start weight when annealing.")# n_min
    ("--lr"; arg_type=Float64; default=0.001; help="Learning rate")
    ("--min_lr"; arg_type=Float64; default=0.00001; help="Minimum learning rate.")
    ("--lr_decay_rate"; arg_type=Float64; default=0.9999; help="Minimum learning rate.")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
    ("--optimization"; default="Adam(;gclip = 1.0)"; help="Optimization algorithm and parameters.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  kl = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate], o[:kl_tolerance]) #Kullback-Leibler parameters
  lrp = LRparameters(o[:lr], o[:min_lr], o[:lr_decay_rate]) #learning rate parameters
  model = initmodel(o[:enc_rnn_size], o[:dec_rnn_size], o[:V], o[:z_size], o[:num_mixture])
  params = Parameters()
  global optim = initoptim(model, o[:optimization])
  sketchpoints3D, numbatches = loaddata(o[:filename], params)
  trnpoints3D, vldpoints3D, tstpoints3D = splitdata(sketchpoints3D)
  normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params)
  if !o[:readydata]
    savedata(o[:filename], trnpoints3D, vldpoints3D, tstpoints3D)
  end
  trn_batch_count = div(length(trnpoints3D), params.batchsize)
  vld_batch_count = div(length(vldpoints3D), params.batchsize)
  tst_batch_count = div(length(tstpoints3D), params.batchsize)
  trndata, trnseqlens = minibatch(trnpoints3D, trn_batch_count-1, params)
  println("Starting training")
  train(model, trndata, trnseqlens, optim, o[:epochs], lrp, kl)
end
main()
end
