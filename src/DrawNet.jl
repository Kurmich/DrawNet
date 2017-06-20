include("DataLoader.jl")
module DrawNet
using Drawing, DataLoader
using Knet, ArgParse, JLD, AutoGrad
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
function init_s2s_lstm_model(e_H::Int, d_H::Int, V::Int, z_size::Int, num_mixture::Int)
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
  model[:fw_state0] = [initxav(1, H), initxav(1, H)]
  model[:fw_embed] = initxav(V, H) # x = input * model[:fw_embed]; x_dims = [batchsize, H]
  #here x and hidden will be concatenated form lstm_input with dims = [batchsize, H]
  model[:fw_encode] = [ initxav(2H, 4H), initxav(1, 4H) ] #lstm_outdims = [batchsize, H]
  #same analysis goes for the decoder
  model[:bw_state0] = [initxav(1, H), initxav(1, H)]
  model[:bw_embed] = initxav(V, H)
  model[:bw_encode] = [ initxav(2H, 4H), initxav(1, 4H) ] #lstm_outdims = [batchsize, H]
end

#=
model -> rnn model
e_H -> size of hidden state of the encoder
d_H -> size of hidden state of the decoder
z_size -> size of latent vector z
=#
function initpredecoder(model, e_H::Int, d_H::Int, z_size::Int)
  #Incoming input dims = [batchsize, 2e_H]
  model[:mu] = [initxav(2e_H, 1), initxav(1, 1)] #mu = input * W_mu .+ b_mu -> dims = [batchsize, 1]
  model[:sigma] = [initxav(2e_H, 1), initxav(1, 1)] #sigma = input * W_sigma .+ b_sigma -> dims = [batchsize, 1]
  #perform z = mu .+ sigma*N(0, I) -> z_dims = [batchsize, z_size]
  model[:z] = [initxav(z_size, 2d_H), initxav(1, 2d_H)] # dec_H_0 = z*W_z .+ b_z -> dims = [batchsize, d_H]
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
  model[:decode] = [ initxav(2H, 4H), initxav(1, 4H) ] #lstm_outdims = [batchsize, H]
  ##model[:output] = [initxav(H, 6num_mixture + 3 ), initxav(1, 6num_mixture + 3 )] #output = lstm_out * W_output .+ b_output -> dims = [batchsize, 6*num_mixture + 3]
  #mixture weights
  model[:out_p] = [initxav(H, num_mixture), initxav(1, num_mixture)] #output = lstm_out * W_out_p .+ b_out_p -> dims = [batchsize, num_mixture]
  #3 logits
  model[:out_q] = [initxav(H, 3), initxav(1, 3)] #output = lstm_out * W_out_q .+ b_out_q -> dims = [batchsize, 3]
  #distribution parameters 5 for each mixture
  model[:out_params] = [initxav(H, 5num_mixture), initxav(1, 5num_mixture)] #output = lstm_out * W_out_params .+ b_out_params -> dims = [batchsize, 5*num_mixture]
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
  prob = exp( -( z/(2*t) ) ) / (2*pi*sigma_x*sigma_y*t)
  return prob
end
function softmax(p, d)
  tmp = exp(p)
  return tmp ./ sum(tmp, d)
end
function s2s(model, data, seqlen, wkl; epsilon = 1e-6)
  #model settings
  maxlen = maximum(seqlen)
  num_mixture = size(model[:out_p][1], 2)
  V = size(data[1], 2)
  batchsize = size(data[1], 1)
  #println("batchsize = $(batchsize)")
  d_H = size(model[:embed], 2)
  z_size = size(model[:z][1], 1)
  #this ones are wrong
  statefw = initstate(batchsize, model[:fw_state0])
  #println(size(statefw[1]))
  statebw = initstate(batchsize, model[:bw_state0])
  #forward encoder
  for i = 1:maxlen
    input = data[i] * model[:fw_embed]
    statefw = lstm(model[:fw_encode], statefw, input)
  end
  #backward encoder
  for i=maxlen:-1:1
    input = data[i]*model[:bw_embed]
    statebw = lstm(model[:bw_encode], statebw, input)
  end

  #predecoder step
  (h_fw, c_fw) = statefw
  (h_bw, c_bw) = statebw
  #println("h_fw[1, 1] = $(h_fw[1, 1])")
  h = hcat(h_fw, h_bw)
  mu = h*model[:mu][1] .+ model[:mu][2]
  #TO DO: DIMANESIONS OF SIGMA AND GAUSSIAN
  sigma_cap = h*model[:sigma][1] .+ model[:sigma][2]
  sigma = exp( sigma_cap/2 )
  #println("sizesigma = $(size(sigma)) sizemu= $(size(mu))")
  z = mu .+ sigma .* atype(gaussian(batchsize, z_size;mean=0, std=1))

  #decoder step
  hc = tanh(z*model[:z][1] .+ model[:z][2])
  #println(size(hc))
  state = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
  penstate_loss = 0
  offset_loss = 0
  for i=2:maxlen
    #dims data = [batchsize, V] = [batchsize, 5]
  #  println("data[i-1][1,1] $(data[i-1][1,1])")
    input = hcat(data[i-1], z) #concatenate latent vector with previous point
    input = input * model[:embed]
    state = lstm(model[:decode], state, input)
    #Calculate L_s on paper
    #result1 = -tf.log(result1 + epsilon) # avoid log(0)
    mix_coeff = predict(model[:out_p], state[1])
  #  println("mix coeff 1 1 $(mix_coeff[1, 1])")
    pnorm = softmax(mix_coeff, 2)
  #  println("size of pnorm $(size(pnorm))")
    mix_params = predict(model[:out_params], state[1]) #are parameters different for each sketch in batch? or same parameters for single batch? one for all seems reasonable CHANGE THIS
  #  println("size of mix_params $(size(mix_params))")
    #iterate over all mixtures
    for j = 1:batchsize
      mix_probs = 0
      for m = 1:num_mixture
        #are mix params different for each sketch point in batch ? NEED TO CLARIFY THIS.
        #result1 = -tf.log(result1 + epsilon) # avoid log(0)
        mixIdx = 5*(m-1)
  #      println("mix weight $(AutoGrad.getval(pnorm[j, m]))")
        mix_probs += pnorm[j, m] * bivariate_prob(data[i][j, 1], data[i][j, 2], mix_params[j, mixIdx+1], mix_params[j, mixIdx+2], exp(mix_params[j, mixIdx+3]), exp(mix_params[j, mixIdx+4]), tanh(mix_params[j, mixIdx+5])) ##vector scalar ? handle this part ?
      end
  #    println("sum of mix probs $(AutoGrad.getval(mix_probs))")
      offset_loss += -log(mix_probs + epsilon)
    end

    #Calculate L_p in paper
    logits = predict(model[:out_q], state[1])
    qnorm = logp(logits, 2)
  #  println("size of qnorm $(size(qnorm))")
  #  println("size of data[i] $(size(data[i])) ")
    penstate_loss += -sum(data[i][:, (V-2):V] .* qnorm)
  end
  kl_loss = -sum((1 + sigma_cap + mu.*mu - exp(sigma_cap))) / (2*z_size*batchsize)
  offset_loss /= (maxlen * batchsize)
  penstate_loss /= (maxlen * batchsize)
  #println("size kl_loss = $(size(kl_loss)) size offset_loss = $(size(offset_loss)) size penstate_loss = $(size(penstate_loss))")
  loss = offset_loss + penstate_loss + wkl*kl_loss
  println("loss = $(AutoGrad.getval(loss)) ")
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

s2sgrad = grad(s2s)
function train(model, data, seqlens, wkl, opts, epochs)
  for e = 1:epochs
    for i = 1:length(data)
      grads = s2sgrad(model, map(a->convert(atype, a), data[i]), seqlens[i], wkl)
      update!(model, grads, opts)
    end
  end
end

function avgloss(model, data, seqlen, wkl)
  sumloss = 0
  for i = 1:length(data)
    sumloss += s2s(model, atype(data[i]), seqlen[i], wkl)
  end
  return sumloss/length(data)
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


# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a)
initoptim(a,otype)=map(x->initoptim(x,otype), a)

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="A Neural Representation of Sketch Drawings. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--epochs"; arg_type=Int; default=10; help="Total number of training set. Keep large.")
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
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
    ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  model = init_s2s_lstm_model(o[:enc_rnn_size], o[:dec_rnn_size], o[:V], o[:z_size], o[:num_mixture])
  params = Parameters()
  global optim = initoptim(model, o[:optimization])
  sketchpoints3D, numbatches = loaddata(o[:filename], params)
  data, seqlens = minibatch(sketchpoints3D, 3, params)
  info("Starting training")
  train(model, data, seqlens, 1, optim, o[:epochs])
end
main()
end
