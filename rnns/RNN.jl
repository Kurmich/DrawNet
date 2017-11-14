
global const pretrnp = "../pretrained/"
global const datap = "../data/"
global const annotp = "../annotateddata/"
global const atype = ( gpu() >= 0 ? KnetArray{Float32} : Array{Float32} )
initxav(d...) = atype(xavier(d...))
initzeros(d...) = atype(zeros(d...))
initones(d...) = atype(ones(d...))
initrandn(winit=0.0001, d...) = atype(winit*randn(d...))



function reportmodel(model, o)
  M = Int((size(model[:output][1], 2) - (o[:V]-2) )/6) #number of mixtures
  d_H = size(model[:output][1], 1) #decoder hidden unit size
  e_H = size(model[:fw_state0][2], 2)
  z_size = size(model[:sigma_cap][1], 2) #size of latent vector z
  @printf("Num of mixtures: %d, num of encoder units: %d , num of decoder units: %d, latent vector size: %d \n", M, e_H, d_H, z_size)
end


function initransfer(o)
  model = Dict{Symbol, Any}()
  e_H, d_H = o[:enc_rnn_size], o[:dec_rnn_size]
  numclasses = o[:numclasses]
  z_size = o[:z_size]
  model[:w1] = [initxav(4d_H, 2d_H), initzeros(1, 2d_H) ]
  model[:w2] = [initxav(2d_H, 2d_H), initzeros(1, 2d_H) ]
  model[:pred] = [initxav(2d_H, numclasses), initzeros(1, numclasses)]
  return model
end

function init_con_segmenter( o )
  e_H, d_H = o[:enc_rnn_size], o[:dec_rnn_size]
  V, z_size, num_mixture = o[:V], o[:z_size], o[:num_mixture]
  numclasses = o[:numclasses]
  model = Dict{Symbol, Any}()
  info("Initializing encoder.")
  initencoder(model, e_H, V, 0)
  info("Encoder was initialized. Initializing predecoder.")
  initpredecoder(model, e_H, d_H, z_size, 0)
  info("Predecoder was initialized. Initializing decoder.")
  initsegdecoder(model, d_H, V, num_mixture, z_size, 0)
  info("Decoder was initialized. Initializing shifts.")
  initshifts(model, e_H, d_H, z_size)
  info("Initialization complete.")
  init_con_predictor(model, d_H, numclasses)
  if o[:attn]
    initattention(model, e_H)
  end
  return model
end

function initsegdecoder(model, H::Int, V::Int, num_mixture::Int, z_size::Int, imlen::Int)
  #incoming input dims = [batchsize, z_size + V]
  #model[:state0] = [initxav(1, H), initzeros(1, H)]
  model[:embed] = initxav(V + z_size, H) # x = input * model[:embed]; x_dims = [batchsize, H]
  model[:decode] = [ initxav(H + H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
end

function init_con_predictor(model, d_H, numclasses)
  #input dims = [batchsize, H]
  model[:pred] = [initxav(d_H, numclasses), initzeros(1, numclasses)]
end

function initsegmenter( o )
  #initial hidden and cell states of forward encoder
  e_H, numclasses = o[:enc_rnn_size], o[:numclasses]
  d_H = o[:dec_rnn_size]
  e_H = d_H
  V, z_size, num_mixture = o[:V], o[:z_size], o[:num_mixture]
  model = Dict{Symbol, Any}()
  if !o[:hascontext]
    z_size = 0
  else
    model[:z] = [initxav(z_size, 2e_H), initzeros(1, 2e_H)]
  end

  info("Initializing encoder.")
  initencoder(model, e_H, V, z_size)
  initpredictor(model, e_H, numclasses)
  if o[:attn]
    initattention(model, e_H)
  end
  model[:fw_shifts] = getshifts(e_H)
  model[:bw_shifts] = getshifts(e_H)
  return model
end

function initattention(model, e_H)
  model[:fwattn] = [initxav(e_H, 1), initzeros(1, 1)]
  model[:bwattn] = [initxav(e_H, 1), initzeros(1, 1)]
end


function initpredictor(model, e_H, numclasses)
  #input dims = [batchsize, H]
  model[:pred] = [initxav(2e_H, numclasses), initzeros(1, numclasses)]
end


function initcontextmodel( o )
  e_H, d_H = o[:enc_rnn_size], o[:dec_rnn_size]
  V, z_size, num_mixture = o[:V], o[:z_size], o[:num_mixture]
  imlen = 0
  model = Dict{Symbol, Any}()
  initencoder(model, e_H, V, imlen)
  model[:fw_output] = [ initxav(e_H, 6num_mixture + V - 2 ), initzeros(1, 6num_mixture + V - 2 ) ]
  model[:bw_output] = [ initxav(e_H, 6num_mixture + V - 2 ), initzeros(1, 6num_mixture + V - 2 ) ]
  model[:fw_shifts] = getshifts(e_H)
  model[:bw_shifts] = getshifts(e_H)
  return model
end

#=
e_H -> size of hidden state of the encoder
d_H -> size of hidden state of the decoder
z_size -> size of latent vector z
V -> point vector size (i.e. 5 for (delta_x, delta_y, p1, p2, p3))
num_mixture -> number of gaussian mixtures
=#
function initmodel( o )
  #initial hidden and cell states of forward encoder
  e_H, d_H = o[:enc_rnn_size], o[:dec_rnn_size]
  V, z_size, num_mixture = o[:V], o[:z_size], o[:num_mixture]
  imlen = 0
  model = Dict{Symbol, Any}()
  info("Initializing encoder.")
  initencoder(model, e_H, V, imlen)
  info("Encoder was initialized. Initializing predecoder.")
  initpredecoder(model, e_H, d_H, z_size, imlen)
  info("Predecoder was initialized. Initializing decoder.")
  initdecoder(model, d_H, V, num_mixture, z_size, imlen)
  info("Decoder was initialized. Initializing shifts.")
  initshifts(model, e_H, d_H, z_size)
  info("Initialization complete.")
  return model
end

function initshifts(model, e_H, d_H, z_size)
  model[:fw_shifts] = getshifts(e_H)
  model[:bw_shifts] = getshifts(e_H)
  model[:dec_shifts] = getshifts(d_H)
end

function getshifts(h::Int)
  alpha = initones(1, h)
  beta  = initzeros(1, h)
  return [ alpha, beta ]
end

#=
model -> rnn model
H -> size of hidden state of the encoder
V -> point vector size (i.e. 5 for (delta_x, delta_y, p1, p2, p3))
=#
function initencoder(model, H::Int, V::Int, z_size::Int)
  #incoming input -> dims = (batchsize, V=5)
  model[:fw_state0] = [initxav(1, H), initzeros(1, H)]
  model[:fw_embed] = initxav(V + z_size, H) # x = input * model[:fw_embed]; x_dims = [batchsize, H]
  #here x and hidden will be concatenated form lstm_input with dims = [batchsize, H]
  model[:fw_encode] = [ initxav(2H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
  #same analysis goes for the decoder
  model[:bw_state0] = [initxav(1, H), initzeros(1, H)]
  model[:bw_embed] = initxav(V+z_size, H)
  model[:bw_encode] = [ initxav(2H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
end

#=
model -> rnn model
e_H -> size of hidden state of the encoder
d_H -> size of hidden state of the decoder
z_size -> size of latent vector z
=#
function initpredecoder(model, e_H::Int, d_H::Int, z_size::Int, imlen::Int)
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
function initdecoder(model, H::Int, V::Int, num_mixture::Int, z_size::Int, imlen::Int)
  initxav(d...) = atype(xavier(d...))
  #incoming input dims = [batchsize, z_size + V]
#  model[:embed] = initxav(z_size + V, H) # x = input * model[:embed]; x_dims = [batchsize, H]
  model[:decode] = [ initxav(z_size + V + H, 4H), initzeros(1, 4H) ] #lstm_outdims = [batchsize, H]
  model[:output] = [ initxav(H, 6num_mixture + V - 2 ), initzeros(1, 6num_mixture + V - 2 ) ] #output = lstm_out * W_output .+ b_output -> dims = [batchsize, 6*num_mixture + 3]
end


#MODELS

function vanilla_lstm(param, state, input; dprob=0, exposed::Bool=false)
  weight, bias = param
  hidden, cell = state
  h = size(hidden, 2)
  gates = hcat(input, hidden) * weight .+ bias
  forget = sigm(gates[:, 1:h])
  ingate = sigm(gates[:, 1+h:2h])
  outgate = sigm(gates[:, 1+2h:3h])
  change = tanh(gates[:, 1+3h:4h])
  cell = cell .* forget + ingate .* dropout(change, dprob) #memoryless dropout
  hidden = outgate .* tanh(cell)
  if exposed #return all gates if needed for inspection
    return (hidden, cell), (forget, ingate, outgate, change)
  end
  return (hidden, cell)
end

function normlayer(x, alpha, beta; epsilon=1e-3, mean = nothing, rstd = nothing) #speedup with precalculated mean and std?
  #=
  dims(x) = (batchsize, hiddensize)
  dims(alpha) = dims(beta) = (1, hiddensize)
  dims(mean) = dims(std) = (batchsize, 1)
  =#
  mean = sum(x, 2)/size(x, 2) #dims = [batchsize, 1] sum over hidden units
  x_shifted = x .- mean
  var = sum(x_shifted.*x_shifted, 2)/size(x_shifted, 2) #dims = [batchsize, 1] sum over hidden units
  rstd = 1 ./ sqrt(var + epsilon)
  return alpha .* x_shifted .* rstd .+ beta
end



function layernorm_lstm(param, state, input, alpha, beta; dprob=0, exposed::Bool=false)
  weight, bias = param
  hidden, cell = state
  h = size(hidden, 2)
  gates = hcat(input, hidden) * weight .+ bias
  forget = sigm(normlayer(gates[:, 1:h], alpha[1], beta[1]))
  ingate = sigm(normlayer(gates[:, 1+h:2h], alpha[2], beta[2]))
  outgate = sigm(normlayer(gates[:, 1+2h:3h], alpha[3], beta[3]))
  change = tanh(normlayer(gates[:, 1+3h:4h], alpha[4], beta[4]))
  cell = cell .* forget + ingate .* dropout(change, dprob) #memoryless dropout
  hidden = outgate .* tanh(normlayer(cell, alpha[5], beta[5]))
  if exposed #return all gates if needed for inspection
    return (hidden, cell), (forget, ingate, outgate, change)
  end
  return (hidden, cell)
end


function lstm(param, state, input; alpha = nothing, beta = nothing, dprob=0, exposed::Bool=false)
  if alpha == nothing && beta == nothing
    vanilla_lstm(param, state, input; dprob=dprob, exposed=exposed)
  elseif alpha != nothing && beta != nothing
    layernorm_lstm(param, state, input, alpha, beta; dprob=dprob, exposed=exposed)
  else
    error("alpha and beta must both be initialized or uninitialized")
  end
end
