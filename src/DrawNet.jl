include("../utils/DataLoader.jl")
include("../idm/IDM.jl")
module DrawNet
using Drawing, DataLoader, IDM, JSON
using Knet, ArgParse, JLD, AutoGrad
include("../rnns/RNN.jl")
include("../models/StrokeRNN.jl")
include("DataManager.jl")

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
  prob = exp.(-(z ./ (2.*neg_rho)) ) ./ (2*pi.*s1s2.*sqrt.(neg_rho))
  return prob
end

function softmax(p, d::Int)
  tmp = exp.(p)
  return tmp ./ sum(tmp, d)
end

function initstate(batchsize, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), batchsize, length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), batchsize, length(c)), 0)
    return (h,c)
end

function encode(model, data, seqlens, batchsize::Int; dprob = 0, meanrep::Bool = false, attn::Bool = false, istate = nothing)
  #Initialize states for forward-backward rnns
  maxlen = length(data) #maximum length of the input sequence
  #println("encode maxlen ", maxlen)
#  statefw = initstate(batchsize, model[:fw_state0])
#  statebw = initstate(batchsize, model[:bw_state0])
  if istate != nothing
    statefw = istate
    statebw = istate
  else
    statefw = initstate(batchsize, model[:fw_state0])
    statebw = initstate(batchsize, model[:bw_state0])
  end
  fwa, bwa = nothing, nothing
  if meanrep || attn
    fwmean = atype(zeros(size(statefw[1])))
    bwmean = atype(zeros(size(statebw[1])))
  end
  if attn
    fwm = []
    bwm = []
  end
  #forward encoder
  w = AutoGrad.getval(model)
  hasembed, hasshift = haskey(w, :fw_embed), haskey(w, :fw_shifts)
  alpha, beta  = hasshift ? (model[:fw_shifts][1], model[:fw_shifts][2]) : (nothing, nothing)
  for i = 1:maxlen
    #println(size(data[i]), size(model[:fw_embed]))
    input = hasembed ? data[i] * model[:fw_embed] : data[i]
    statefw = lstm(model[:fw_encode], statefw, input; alpha=alpha, beta=beta, dprob=dprob)
    if meanrep
      fwmean += statefw[1]
    elseif attn
      push!(fwm, statefw[1])
      if i == 1
        fwa = statefw[1]*model[:fwattn][1] .+ model[:fwattn][2]
      else
        fwa = hcat(fwa, statefw[1]*model[:fwattn][1] .+ model[:fwattn][2])
      end
    end
  end
  #backward encoder
  hasembed, hasshift = haskey(w, :bw_embed), haskey(w, :bw_shifts)
  alpha, beta  = hasshift ? (model[:bw_shifts][1], model[:bw_shifts][2]) : (nothing, nothing)
  for i = maxlen:-1:1
    input = hasembed ? data[i]*model[:bw_embed] : data[i]
    statebw = lstm(model[:bw_encode], statebw, input; alpha=alpha, beta=beta, dprob=dprob)
    if meanrep
      bwmean += statebw[1]
    elseif attn
      push!(bwm, statebw[1])
      if i == maxlen
        bwa = statebw[1]*model[:bwattn][1] .+ model[:bwattn][2]
      else
        bwa = hcat(bwa, statebw[1]*model[:bwattn][1] .+ model[:bwattn][2])
      end
    end
  end

  if meanrep
    fwmean = fwmean ./ maxlen
    bwmean = bwmean ./ maxlen
    return hcat(fwmean, bwmean)
  elseif attn
    bwa = softmax(bwa, 2)
    fwa = softmax(fwa, 2)
    for i = 1:maxlen
      fwmean += (fwm[i] .* fwa[:, i])
      bwmean += (bwm[i] .* bwa[:, i])
    end
    return hcat(fwmean, bwmean)
  end
  return hcat(statefw[1], statebw[1]) #(h_fw, c_fw) = statefw, (h_bw, c_bw) = statebw
end


function get_mixparams(output, M::Int, V::Int; samplemode=false)
  #Here I used different ordering for outputs; in practice order doesn't matter
  pnorm = softmax(output[:, 1:M], 2) #normalized distribution probabilities
  mu_x = output[:, M+1:2M]
  mu_y = output[:, 2M+1:3M]
  sigma_x = exp.(output[:, 3M+1:4M])
  sigma_y = exp.(output[:, 4M+1:5M])
  rho = tanh.(output[:, 5M+1:6M])
  if samplemode
    qnorm = softmax(output[:, 6M+1:6M+(V-2)], 2) #normalized log probabilities of logits
  else
    qnorm = logp(output[:, 6M+1:6M+(V-2)], 2) #normalized log probabilities of logits
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
  batchsize = size(inputpoints[1], 1)
  z_size = size(model[:z][1], 1) #size of latent vector z
  h = encode(model, inputpoints, seqlen, batchsize)
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

function appendlatentvec(data, z)
  adata = []
  for i=1:length(data)
    result = hcat(data[i], z)
    push!(adata, result)
  end
  return adata
end

#random scaling of x and y values
function perturb(data; scalefactor=0.1)
  pdata = []
  for i=1:length(data)
    x_scalefactor = (rand() - 0.5) * 2 * scalefactor + 1.0
    y_scalefactor = (rand() - 0.5) * 2 * scalefactor + 1.0
    result = deepcopy(data[i])
    result[:, 1] *= x_scalefactor
    result[:, 2] *= y_scalefactor
    #perturb idm if needed
    push!(pdata, result)
  end
  return pdata
end



#update learning rate of parameters
function updatelr!(opts::Associative, cur_lr)
  for (key, val) in opts
    if typeof(val) == Knet.Adam || typeof(val) == Knet.Sgd || typeof(val) == Knet.Rmsprop || typeof(val) == Knet.Momentum || typeof(val) == Knet.Adagrad
      val.lr = cur_lr
    else
      for opt in val
        opt.lr = cur_lr
      end
    end
  end
end

function normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params::Parameters; scalefactor = nothing)
  DataLoader.normalize!(trnpoints3D, params; scalefactor=scalefactor)
  DataLoader.normalize!(vldpoints3D, params; scalefactor=params.scalefactor)
  DataLoader.normalize!(tstpoints3D, params; scalefactor=params.scalefactor)
end

function plainloss(model, data, seqlen, ygold, o; epsilon = 1e-6, istraining::Bool = true, weights = nothing, istate = nothing , z = nothing)
  #model settings
  if z != nothing
    d_H = size(model[:fw_embed], 2)
    #println(d_H)
    data = appendlatentvec(data, z)
    hc = tanh(z * model[:z][1] .+ model[:z][2])
    istate = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
    #println(size(hc))
  end
  if o[:hascontext]
    @assert(istate != nothing)
  end
  if !istraining
    ypred = pred(model, data, seqlen; dprob=0, meanrep=o[:meanrep], attn=o[:attn], istate=istate)
    ynorm = logp(ypred, 2)
    return -sum(ygold .* ynorm)/size(ygold, 1), ypred
  end
  ypred = pred(model, data, seqlen; dprob=o[:dprob], meanrep=o[:meanrep], attn=o[:attn], istate=istate)
  ynorm = logp(ypred, 2)
  if weights == nothing
    return -sum(ygold .* ynorm)/size(ygold, 1)
  end
  return -sum(ygold .* ynorm .* weights)/size(ygold, 1)
end

function pred(model, data, seqlens; dprob = 0, meanrep::Bool = false, attn::Bool = false, istate = nothing )
  (batchsize, V) = size(data[1])
  h = encode(model, data, seqlens, batchsize; dprob=dprob, meanrep=meanrep, attn=attn, istate=istate)
  #LOOK AT INDICES OF LOGP AND SIZE AND SUM
  ypred = h*model[:pred][1] .+ model[:pred][2]
end
function countcorrect(ypred, ygold, correct_count, instance_count)
  #=size(ygold) = (batchsize, numclasses)=#
  correct_batch = sum(ygold .* (ypred .== maximum(ypred, 2)), 1) #dims = 1, numclasses
  instance_batch = sum(ygold, 1) #dims 1, batchsize
  #println(size(correct_batch), size(correct_count))
  correct_count += correct_batch
  instance_count += instance_batch
  return correct_count, instance_count
	#=correct = sum(ygold .* (ypred .== maximum(ypred, 2)))
	return correct=#
end



function transferloss(model, genmodel, data, seqlen, ygold, o; istraining::Bool = true, weights = nothing, fullgenmodel = nothing, fulldata = nothing, fullseqlen = nothing, avg_idms = nothing)
  seqlen = length(data)
  batchsize = size(data[1], 1)
  if !istraining
    h = encode(genmodel, data, seqlen, batchsize; dprob=0)
    if fullgenmodel != nothing
      h1 = encode(fullgenmodel, fulldata, fullseqlen, batchsize; dprob=0)
      h = hcat(h, h1)
    end
    if avg_idms != nothing
      h = hcat(h, avg_idms)
    end
    h = relu.(h*model[:w1][1] .+ model[:w1][2])
    h = relu.(h*model[:w2][1] .+ model[:w2][2])
    ypred = h*model[:pred][1] .+ model[:pred][2]
    ynorm = logp(ypred, 2)
    return -sum(ygold .* ynorm)/size(ygold, 1), ypred
  end
  h = encode(genmodel, data, seqlen, batchsize; dprob=o[:dprob])
  if fullgenmodel != nothing
    h1 = encode(fullgenmodel, fulldata, fullseqlen, batchsize; dprob=o[:dprob])
    h = hcat(h, h1)
  end
  if avg_idms != nothing
    h = hcat(h, avg_idms)
  end
  h = relu.(h*model[:w1][1] .+ model[:w1][2])
  h = dropout(h, 0.5)
  h = relu.(h*model[:w2][1] .+ model[:w2][2])
  h = dropout(h, 0.5)
  ypred = h*model[:pred][1] .+ model[:pred][2]
  ynorm = logp(ypred, 2)
  if weights == nothing
    return -sum(ygold .* ynorm)/size(ygold, 1)
  end
  return -sum(ygold .* ynorm .* weights)/size(ygold, 1)
end


function evalsegm(model, data, f_data, seqlens, ygold, o; genmodel = nothing, fullgenmodel = nothing, f_seqlens= nothing, weights = nothing, idms = nothing)
  correct_count = zeros(1, size(ygold[1], 2))
  instance_count = zeros(1, size(ygold[1], 2))
  curloss, curright = 0.0, 0.0
  loss, correct = 0.0, 0.0
  ygold = map(a->convert(atype, a), ygold)
  count = 0.0
  for i = 1:length(data)
    if o[:hascontext]
      #d_H = size(genmodel[:output][1], 1)
      x = map(a->convert(atype, a), data[i])
      xfull = map(a->convert(atype, a), f_data[i])
      if idms != nothing
        avg_idms = atype(idms[i])
      else
        avg_idms = nothing
      end
      curloss, ypred = transferloss(model, genmodel, x, seqlens[i], ygold[i], o; istraining = false, weights=weights, fullgenmodel=fullgenmodel, fulldata=xfull, fullseqlen = f_seqlens[i], avg_idms=avg_idms)
    else
      curloss, ypred = plainloss(model, map(a->convert(atype, a), data[i]), seqlens[i], ygold[i], o; istraining=false, weights=weights)
    end
    correct_count, instance_count = countcorrect(Array(ypred), Array(ygold[i]), correct_count, instance_count)
    loss += curloss
    count += size(ygold[i], 1) #CHECK
  end
  return loss/count, correct_count, instance_count
end

gradtrans = grad(transferloss)
gradplain = grad(plainloss)
function segment(model, dataset, opts, o, tstaccs; genmodel = nothing, fullgenmodel = nothing)
  (trndata, trnseqlens, trngold, trnstats, f_trndata, f_trnseqlens, trnavgidms)  = dataset[:trn]
  (vlddata, vldseqlens, vldgold, vldstats, f_vlddata, f_vldseqlens, vldavgidms) = dataset[:vld]
  (tstdata, tstseqlens, tstgold, tststats, f_tstdata, f_tstseqlens, tstavgidms) = dataset[:tst]
  append!(trndata, vlddata)
  append!(trnseqlens, vldseqlens)
  append!(trngold, vldgold)
  append!(trnavgidms, vldavgidms)
  #append!(trndata, tstdata)
  #append!(trnseqlens, tstseqlens)
  #append!(trngold, tstgold)
  append!(f_trndata, f_vlddata)
  append!(f_trnseqlens, f_vldseqlens)

  trnstats += vldstats
  (vlddata, vldseqlens, vldgold, vldstats, f_vlddata, f_vldseqlens, vldavgidms) = (tstdata, tstseqlens, tstgold, tststats, f_tstdata, f_tstseqlens, tstavgidms)
  f_vlddata = f_tstdata
  println(trnstats)
  weights = -trnstats/maximum(trnstats)
  weights[weights.==0] = -10
  println(weights)
  weights = softmax(weights, 2) # per class weights for loss function
  println(weights)
  flush(STDOUT)
  weights = atype(weights)
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  trngold = map(a->convert(atype, a), trngold)
  @assert(length(trndata) == length(trngold))
  bestacc = 0
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      cur_wkl = KL.w - (KL.w - KL.wstart) * ((KL.decayrate)^step)
      cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
      x = perturb(trndata[i]; scalefactor=o[:scalefactor])
      xfull = perturb(f_trndata[i]; scalefactor=o[:scalefactor])
      if o[:hascontext]
        x = map(a->convert(atype, a), x)
        xfull = map(a->convert(atype, a), xfull)
        avg_idms = atype(trnavgidms[i])
        avg_idms = nothing
        grads = gradtrans(model, genmodel, x, trnseqlens[i], trngold[i], o; istraining = true, weights=weights, fullgenmodel=fullgenmodel, fulldata=xfull, fullseqlen = f_trnseqlens[i], avg_idms=avg_idms)
      else
        grads = gradplain(model, map(a->convert(atype, a), x), trnseqlens[i], trngold[i], o; weights=weights)
      end
      updatelr!(opts, cur_lr)
      update!(model, grads, opts)
      step += 1
    end
    if o[:a_datasize] <= 400 && e%20 != 0
      continue
    end
    vld_loss, correct_count, instance_count = evalsegm(model, vlddata, f_vlddata, vldseqlens, vldgold, o; genmodel=genmodel, fullgenmodel=fullgenmodel, f_seqlens =f_vldseqlens, weights=weights, idms = nothing)
    #save the best model
    if vld_loss < best_vld_cost
      best_vld_cost = vld_loss
      arrmodel = convertmodel(model)
      println("Epoch: $(e) saving best model to $(pretrnp)$(o[:bestmodel])")
      println("CHECK SCALEFACTOR")
      save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
    end
    #report losses
    curacc = sum(correct_count)/sum(instance_count)
    bestacc = max(curacc, bestacc)
    @printf("vld data - epoch: %d step: %d lr: %g wkl vld loss: %g total acc: %g\n", e, step, cur_lr, vld_loss, curacc)
    for c = 1:length(correct_count)
      @printf("vld data - epoch: %d; class %d; instances: %d; correct instances: %d; acc: %g \n", e, c, instance_count[c], correct_count[c], correct_count[c]/instance_count[c] )
    end
    #=save every o[:save_every] epochs
    if e%o[:save_every] == 0
      arrmodel = convertmodel(model)
      save("$(pretrnp)m$(e)$(o[:tmpmodel])","model", arrmodel)
    end=#
    flush(STDOUT)
  end
  push!(tstaccs, bestacc)
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


function getstrokeseqs4d(x_batch_5D)
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
  V = 4
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
        apstroke[1][j, 4] = 1
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
          stroke[k][j, :] = [0 0 0 1]
        end
        continue
      end
      s = stroke_start[j] #starting index i'th stroke of j'th sketch
      e = end_indices[j][i] #ending index of i'th stroke of j'th sketch
      #init points
      for k = s:e
      #  println("$(size(stroke[k-stroke_start[j]+1][j, :]))  $(size(x5D[k, :]))")
        stroke[k-s+1][j, 1:2] = x5D[k, 1:2]
        stroke[k-s+1][j, 3] = 1
      end
      #pad ends if needed
      for k = (e-s+2):length(stroke)
        stroke[k][j, :] = [0 0 0 1]
      end
      #DONOT FORGET TO MODIFY START[j]
      stroke_start[j] = end_indices[j][i] + 1
    end

    if i > 1 #if not first stroke then add [0,0,1,0] at the beginnings
      append!(apstroke, stroke)
      push!(seqlens, max_stroke_len+1)
    else #for first [0,0,1,0] already added
      apstroke = stroke
      push!(seqlens, max_stroke_len)
    end
    #push!(seqlens, max_stroke_len+1)
    push!(stroke_batch, apstroke)
  end
  return stroke_batch, seqlens
end


function getdata(filename, a_filename, params)
  return getsketchpoints3D(filename; a_filename = a_filename, params = params)
end

function savedata(filename, trndata, vlddata, tstdata)
  rawname = split(filename, ".")[1]
  save("../data/$(rawname).jld","train", trndata, "valid", vlddata, "test", tstdata)
end

function savedataset(filename, dataset)
  rawname = split(filename, ".")[1]
  save("../data/$(rawname).jld", "dataset", dataset)
end

function loaddataset(filename)
  dataset = load(filename)
  println("loading dataset ",  filename)
  return dataset["dataset"]
end

function loaddata(filename)
  dataset = load(filename)
  println(filename)
  return dataset["train"], dataset["valid"], dataset["test"]
end

function get_splitteddata(data, trnidx, vldidx, tstidx)
  return data[trnidx], data[vldidx], data[tstidx]
end

function makebatches(sketches, points3D, f_sketches, f_points3D, labels, V, params; full::Bool = false)
  #=creates batches of annotated data=#
  @assert(length(sketches)==length(f_sketches), "Error number of sketches is $(length(sketches)) but number of full sketches is $(length(f_sketches))")
  f_data, f_seqlens =  sketch_minibatch(f_points3D, V, params)
  data, seqlens = minibatch(points3D, V, params)
  ygolds, instance_per_label = getlabels(sketches, labels, params)
  println(sketches[1].end_indices)
  println(f_sketches[1].end_indices)
  im = extract_avg_idm(f_sketches[3].points, f_sketches[3].end_indices)
  saveidm(im, "sampleidm.png")
  savesketch(f_sketches[3], "idmsketch.png")
  avg_idms = get_avg_idms(sketches, f_sketches, params)
  @assert(length(ygolds) == length(data))
  return (data, seqlens, ygolds, instance_per_label, f_data, f_seqlens, avg_idms)
end

function makedataset(trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, f_trnsketches, f_trnpoints3D, f_vldsketches, f_vldpoints3D, f_tstsketches, f_tstpoints3D, labels, V, params)
  dataset = Dict()
  dataset[:trn] = makebatches(trnsketches, trnpoints3D, f_trnsketches, f_trnpoints3D, labels, V, params)
  dataset[:vld] = makebatches(vldsketches, vldpoints3D, f_vldsketches, f_vldpoints3D, labels, V, params)
#  bs = params.batchsize
#  params.batchsize = 100
  dataset[:tst] = makebatches(tstsketches, tstpoints3D, f_tstsketches, f_tstpoints3D, labels, V, params)
#  params.batchsize = bs
  return dataset
end


function make_gen_dataset(dict_data, V, params; subsetsize::Int = 0)
  #minibatch
  trnpoints3D, vldpoints3D, tstpoints3D = dict_data[:trn][1], dict_data[:vld][1], dict_data[:tst][1]
  if subsetsize != 0
    println("Using smaller subset of training data for training: size = $(subsetsize)")
    trnpoints3D = trnpoints3D[1:subsetsize]
  end
  trndata, trnseqlens = minibatch(trnpoints3D, V, params)
  vlddata, vldseqlens = minibatch(vldpoints3D, V, params)
  tstdata, tstseqlens = minibatch(tstpoints3D, V, params)
  #normalizeidms!(trnidm, vldidm, tstidm)
  #create dataset
  dataset = Dict()
  dataset[:trn] = (trndata, trnseqlens)
  dataset[:vld] = (vlddata, vldseqlens)
  dataset[:tst] = (tstdata, tstseqlens)
  return dataset
end

function get_gen_data(o, params)
  #=Create dataset for generator model=#
  #= for strokes
  strokedata = getstrokedata(o[:filename]; params=params)
  println("Normalizing stroke data")
  normalizedata!(strokedata[:trn][1], strokedata[:vld][1], strokedata[:tst][1], params)
  println("Is data normalized: $(isnormalized(strokedata[:trn][1]))  $(isnormalized(strokedata[:vld][1]))")
  strokedata[:scalefactor] = params.scalefactor=#

  data = getdata(o[:filename], o[:a_filename], params)
  println("Normalizing stroke data")
  normalizedata!(data[:trn][1], data[:vld][1], data[:tst][1], params)
  println("Is data normalized: $(isnormalized(data[:trn][1]))  $(isnormalized(data[:vld][1]))")
  data[:scalefactor] = params.scalefactor
  #save the data
#  savedata("idx$(o[:filename])", trnidx, vldidx, tstidx)
#  savedata("data$(o[:filename])", trnpoints3D, vldpoints3D, tstpoints3D)
  savedataset("dataset$(o[:filename])", data)
  dataset = make_gen_dataset(data, o[:V], params)
  return dataset
end

function getprocesseddata( dict_data, labels, params)
  #=Input: dict_data - data in a dictionary (label -> list of all strokes with that label)
           labels = list of labels of strokes
  =#
  data = dict2list(dict_data, labels)
  points3D, numbatches, sketches = preprocess(data, params)
  return points3D, numbatches, sketches
end

function makesplit(sketches, labels, o; trn_tst_indices = nothing, trn_vld_indices = nothing, params = nothing)
  rawname = split(o[:a_filename], ".")[1]
  vldsize = 1 / o[:cvfolds]
  if trn_tst_indices == nothing
    trn_tst_indices = randindinces(sketches)
    trndict, tstdict = train_test_split(sketches, vldsize; indices = trn_tst_indices) #get even split as dictionary
    trn_vld_indices = randindinces(trndict)
    trndict, vlddict = train_test_split(trndict, vldsize; indices = trn_vld_indices)
    trntstidxfile, trnvldidxfile  = "$(rawname)trn_tst_indices.jld", "$(rawname)trn_vld_indices.jld"
    println("TRN-VLD-TST split was created and saved to: $(trntstidxfile) $(trnvldidxfile)")
    save(trntstidxfile, "indices" ,trn_tst_indices)
    save(trnvldidxfile, "indices" ,trn_vld_indices)
  else
    println("TRN-VLD-TST split was loaded")
    trndict, tstdict = train_test_split(sketches, vldsize; indices = trn_tst_indices)
    trndict, vlddict = train_test_split(trndict, vldsize; indices = trn_vld_indices)
  end
  println("CHECK ", trn_vld_indices==nothing)
  trnpoints3D, numbatches, trnsketches = getprocesseddata(trndict, labels, params)
  vldpoints3D, numbatches, vldsketches = getprocesseddata(vlddict, labels, params)
  tstpoints3D, numbatches, tstsketches = getprocesseddata(tstdict, labels, params)
  @printf("trnsize: %d vldsize: %d tstsize: %d \n", length(trnpoints3D), length(vldpoints3D), length(tstpoints3D))
  println("IN NORMALIZATION PHASE")
  normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params; scalefactor=48.318977)
  @printf("3D trnsize: %d vldsize: %d tstsize: %d \n", length(trnpoints3D), length(vldpoints3D), length(tstpoints3D))
  return trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, trn_tst_indices, trn_vld_indices
end

function get_seg_data(o, labels; trn_tst_indices = nothing, trn_vld_indices = nothing, params = nothing)
  #=Create dataset for segmentation model=#

  filename = string(annotp, o[:a_filename])
  #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  annotations, annot_dicts = getannotateddata(filename, labels)
  if o[:a_datasize] != 0
    println("Using subset of data, with size $(o[:a_datasize])")
    annotations = getrandannots(annot_dicts, labels, o[:a_datasize])
  end
  sketches, full_sketches = annotated2sketch_obj(annotations)
  for (key, value) in sketches
      println(key, " ==> ", length(value))
  end
  trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, trn_tst_indices, trn_vld_indices = makesplit(sketches, labels, o; trn_tst_indices = trn_tst_indices, trn_vld_indices = trn_vld_indices, params = params)
  f_trnsketches, f_trnpoints3D, f_vldsketches, f_vldpoints3D, f_tstsketches, f_tstpoints3D, f_trn_tst_indices, f_trn_vld_indices = makesplit(full_sketches, labels, o; trn_tst_indices = trn_tst_indices, trn_vld_indices = trn_vld_indices, params = params)
  dataset = makedataset(trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, f_trnsketches, f_trnpoints3D, f_vldsketches, f_vldpoints3D, f_tstsketches, f_tstpoints3D, labels, o[:V], params)
  @printf("# of trn sketches: %d  # of trn batches: %d  \n", length(trnpoints3D), length(f_trnsketches))
  @printf("# of vld sketches: %d  # of vld batches: %d \n", length(vldpoints3D), length(f_vldpoints3D))
  return dataset, annotations, trn_tst_indices, trn_vld_indices
end


function get3ddata(annotations, labels, params)
  sketch_dicts, full_sketch_dicts = annotated2sketch_obj(annotations)
  println("number of sketches = $(length(sketch_dicts)) and number of full sketches $(length(full_sketch_dicts)) ")
  points3D, numbatches, sketches = getprocesseddata(sketch_dicts, labels, params)
  full_points3D, full_numbatches, full_sketches = getprocesseddata(full_sketch_dicts, labels, params)
  return points3D, sketches, full_points3D, full_sketches
end

function get_seg_data2(o, labels; tt = nothing, params = nothing, dpath = annotp, scalefactor = 45, drawing::Bool = false)
  filename = string(dpath, o[:a_filename])
  #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  annotations, annot_dicts = getannotateddata(filename, labels)
  acount = length(annot_dicts)
  tt = (tt == nothing) ? randperm(acount) : tt
  #get training set size for train-test split
  trnsize = acount - div(acount, o[:cvfolds])
  println("Number of annotated data: $(acount) Training set size: $(trnsize)")
  trn_dicts, tst_dicts = data_tt_split(annot_dicts, trnsize; rp = tt)
  if o[:a_datasize] != 0
    trnsize = o[:a_datasize]
    println("Smaller training size $(trnsize)")
    @assert(o[:a_datasize]%o[:cvfolds] == 0, "Annotated training data size must be divisible by $(o[:cvfolds])")
    o[:batchsize] = getnewbatchsize(o[:a_datasize])
    println("New batchsize is $(o[:batchsize])")
    trn_dicts = decrease_trndatasize(trn_dicts, o[:a_datasize], o[:cvfolds])
  end
  tv = randperm(length(trn_dicts))
  #training set size for trainn-valid split
  trnsize = trnsize - div(trnsize, o[:cvfolds])
  trn_dicts, vld_dicts = data_tt_split(trn_dicts, trnsize; rp = tv)
  #get annotation dictionaries label -> list of (points, ends, sketch)
  trnannotations, vldannotations, tstannotations = getannotationdict(trn_dicts, labels), getannotationdict(vld_dicts, labels), getannotationdict(tst_dicts, labels)
  #get 3D points from these annotaions
  trn_points3D, trn_sketches, trn_full_points3D, trn_full_sketches = get3ddata(trnannotations, labels, params)
  vld_points3D, vld_sketches, vld_full_points3D, vld_full_sketches = get3ddata(vldannotations, labels, params)
  tst_points3D, tst_sketches, tst_full_points3D, tst_full_sketches = get3ddata(tstannotations, labels, params)
  @printf("trnsize: %d vldsize: %d tstsize: %d \n", length(trn_points3D), length(vld_points3D), length(tst_points3D))
  println("IN NORMALIZATION PHASE scalefactor is $(scalefactor)")
  normalizedata!(trn_points3D, vld_points3D, tst_points3D, params; scalefactor=scalefactor)
  normalizedata!(trn_full_points3D, vld_full_points3D, tst_full_points3D, params; scalefactor=scalefactor)
  if drawing
    return trn_points3D, vld_points3D, tst_points3D
  end
  dataset = makedataset(trn_sketches, trn_points3D, vld_sketches, vld_points3D, tst_sketches, tst_points3D, trn_full_sketches, trn_full_points3D, vld_full_sketches, vld_full_points3D, tst_full_sketches, tst_full_points3D, labels, o[:V], params)
  return dataset, tt
end

function rnncv(o)
  #=RNN cross validation=#
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  #labels = [ "L", "F", "FP"]
  #labels = ["W", "B", "T", "WNDW", "FA"]
  #labels = [ "EAR", "H", "EYE", "N", "W", "M",  "B", "T", "L"] #for cat
  #labels = [ "LGT", "LDR", "B", "C", "WNDW", "WHS",  "WHL"]
  #labels = [ "B", "S", "L"] #for chair
  labels = [ "P", "C" ,"S", "L"] #for flower
  #labels = ["body", "wing", "horistab", "vertstab",  "engine", "propeller"] #huang airplane
  #labels = ["saddle", "frontframe", "wheel", "handle", "pedal", "chain", "fork", "backframe", "backcover" ] #huang bicycle
  rawname = split(o[:a_filename], ".")[1]
  #gendataset = loaddataset("$(datap)dataset$(o[:dataset])")
  #scalefactor = 48.290142 #firetruck
  #scalefactor = 56.090145 #chair
  #scalefactor = 31.883362 #flower
  #scalefactor = 43.812866 #airplane
  scalefactor = 49.193924 #cat
  #scalefactor = gendataset[:scalefactor]
  #gendataset = nothing
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
  o[:numclasses] = length(labels)
  model = initransfer(o) #initsegmenter(o)
  params = Parameters()
  params.batchsize = o[:batchsize]
  params.min_seq_length = 1
  global optim = initoptim(model, o[:optimization])
  vldsize = 1 / o[:cvfolds]
  smooth = true
  params.max_seq_length = 200
  params.min_seq_length = -1

  if !o[:readydata]
    dataset, tt = get_seg_data2(o, labels; params=params, dpath = dpath, scalefactor=scalefactor)
    save("annotsplits/$(rawname)indices.jld", "indices", tt)
  else
    #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
    tt = load("annotsplits/$(rawname)indices.jld")["indices"]
    println("Fold $(o[:fold])")
    for i=1:(o[:fold]-1)
      println("Shifting $(i)")
      tt = getshiftedindx(tt, o)
    end
    params.max_seq_length = 200
    dataset, tt = get_seg_data2(o, labels; tt = tt, params=params, dpath = dpath, scalefactor=scalefactor)
    #dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:a_filename], labels, vldsize; trn_tst_indices=trn_tst_indices, trn_vld_indices=trn_vld_indices, params=params)
  end
  println("Starting training")
  flush(STDOUT)
  genmodel = nothing
  fullgenmodel = nothing
  if o[:hascontext]
    println("Loading context model from $(pretrnp)$(o[:model])")
    w = load("$(pretrnp)$(o[:model])")
    genmodel = revconvertmodel(w["model"])
    #println("Loading full context model from $(pretrnp)$(o[:fullmodel])")
  #  w = load("$(pretrnp)$(o[:fullmodel])")
  #  fullgenmodel = revconvertmodel(w["model"])
  else
    genmodel = nothing
    fullgenmodel = nothing
  end
  tstaccs = Float64[]
  segment(model, dataset, optim, o, tstaccs; genmodel=genmodel, fullgenmodel=fullgenmodel)
end


function segmentation_mode(o)
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  #labels = [ "L", "F", "FP"]
  #labels = ["W", "B", "T", "WNDW", "FA"]
  #labels = [ "EAR", "H", "EYE", "N", "W", "M",  "B", "T", "L"] #Labels for cat
  labels = [ "LGT", "LDR", "B", "C", "WNDW", "WHS",  "WHL"] #labels for firetruck
  repeat = 20
  o[:numclasses] = length(labels)
  model = initransfer(o)
  params = Parameters()
  params.batchsize = o[:batchsize]
  params.min_seq_length = 1
  global optim = initoptim(model, o[:optimization])
  vldsize = 1 / o[:cvfolds]
  smooth = true
  rawname = split(o[:a_filename], ".")[1]
  params.max_seq_length = 200
  if !o[:readydata]
    #dataset, annotations, trn_tst_indices, trn_vld_indices = get_seg_data(o, labels; params=params)
    alldata = []
    for i=1:repeat
      params.max_seq_length = 200
      dataset, tt = get_seg_data2(o, labels; params=params)
      push!(alldata, (dataset, tt) )
    end
  else
    #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
    fln = "annotsplits/$(rawname)$(o[:a_datasize])$(o[:fold]).jld"
    println("Loading data from $(fln)")
    d = load(fln)
    alldata = d["data"]
  #  dataset, annotations, trn_tst_indices, trn_vld_indices = get_seg_data(o, labels; trn_tst_indices=trn_tst_indices, trn_vld_indices=trn_vld_indices, params=params)
  end


  fullgenmodel = nothing
  if o[:hascontext]
    println("Loading context model.")
    w = load("$(pretrnp)$(o[:model])")
    genmodel = revconvertmodel(w["model"])
    #w = load("$(pretrnp)$(o[:fullmodel])")
    #fullgenmodel = revconvertmodel(w["model"])
  else
    genmodel = nothing
  end
  tstaccs = Float64[]
  if o[:a_datasize] != 0 && !o[:readydata]
    println("Saving data")
    save("annotsplits/$(rawname)$(o[:a_datasize])$(o[:fold]).jld", "data", alldata)
  end
  println("Starting training")
  flush(STDOUT)
  for (dataset, tt) in alldata
    println("Starting new split")
    segment(model, dataset, optim, o, tstaccs; genmodel=genmodel, fullgenmodel=fullgenmodel)
    model = initransfer(o)
    optim = initoptim(model, o[:optimization])
  end
  println(tstaccs)
  println("Mean: $(mean(tstaccs)) STD: $(std(tstaccs))")
end

function generation_mode(o)
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  model = initmodel(o)
  params = Parameters()
  global optim = initoptim(model, o[:optimization])
  if !o[:readydata]
    #create minibatched dataset
    dataset = get_gen_data(o, params)
  else
    #create minibatched dataset from predivided dataset
    println("Loading data for training!")
    dict_data = loaddataset("$(datap)dataset$(o[:dataset])")
    dataset = make_gen_dataset(dict_data, o[:V], params; subsetsize=o[:subsetsize])
  #  trnidm, vldidm, tstidm  = loaddata("$(datap)idm$(o[:imlen])$(o[:dataset])")
  end
  println("Starting training")
  reportmodel(model, o)
#  trndata = paddall(trndata, trnidmtuples, o[:imlen])
  #vlddata = paddall(vlddata, vldidmtuples, o[:imlen])
  #info("padding was complete")
  flush(STDOUT)
  train(model, dataset, optim, o)
end

function context_mode(o)
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  model = initcontextmodel(o)
  params = Parameters()
  global optim = initoptim(model, o[:optimization])
  if !o[:readydata]
    #create minibatched dataset
    dataset = get_gen_data(o, params)
  else
    #create minibatched dataset from predivided dataset
    println("Loading data for training!")
    dict_data = loaddataset("$(datap)dataset$(o[:dataset])")
    dataset = make_gen_dataset(dict_data, o[:V], params)
  #  trnidm, vldidm, tstidm  = loaddata("$(datap)idm$(o[:imlen])$(o[:dataset])")
  end
  println("Starting training")
#  trndata = paddall(trndata, trnidmtuples, o[:imlen])
  #vlddata = paddall(vlddata, vldidmtuples, o[:imlen])
  #info("padding was complete")
  flush(STDOUT)
  contrain(model, dataset, optim, o)
end



function reportparams( o )
  println("Has Attention: $(o[:attn]); Mean Representation: $(o[:meanrep]); GMM Context: $(o[:hascontext])")
  println("Data filename: $(o[:filename]); Annotated data filename: $(o[:a_filename]); Readydata: $(o[:readydata])")
  println("Optimizer: $(o[:optimization]); Vocabsize: $(o[:V]); Annotated datasize(0 for full dataset): $(o[:a_datasize])")
  @printf("lr: %g minlr: %g; lr-decay-rate: %g; dprob: %g; z_size: %g; batchsize: %g \n", o[:lr], o[:minlr], o[:lr_decay_rate], o[:dprob], o[:z_size], o[:batchsize])
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
    ("--save_every"; arg_type=Int; default=1; help="Number of epochs per checkpoint creation.")
    ("--dec_model"; arg_type=String; default="lstm"; help="Decoder: lstm, or ....")
    ("--a_filename"; arg_type=String; default="airplane1014.ndjson"; help="Data file name")
    ("--filename"; arg_type=String; default="airplane.ndjson"; help="Data file name")
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
    ("--lr_decay_rate"; arg_type=Float64; default=0.9999; help="Minimum learning rate.")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
    ("--attn"; action=:store_true; help="true if model has attention")
    ("--meanrep"; action=:store_true; help="true if model uses mean representation")
    ("--segmode"; action=:store_true; help="Store true if in segmentation mode")
    ("--hascontext"; action=:store_true; help="Store true if context info is used")
    ("--conmode"; action=:store_true; help="Store true if context info is used")
    ("--optimization"; default="Adam(;gclip = 1.0)"; help="Optimization algorithm and parameters.")
    ("--dataset"; arg_type=String; default="full_simplified_airplane.jld"; help="Name of the dataset")
    ("--fullmodel"; arg_type=String; default="best_gen_full_airmax50min5.jld"; help="Name of the pretrained model")
    ("--model"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--cvfolds"; arg_type=Int; default=5; help="Number of folds to use for cross validation.")
    ("--fold"; arg_type=Int; default=-1; help="Current fold.")
    ("--a_datasize"; arg_type=Int; default=0; help="Annotated dataset size to use.")
    ("--subsetsize"; arg_type=Int; default=0; help="Dataset size to use for training generative model.")
  end
  println("CHECK SCALEFACTOR")
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  reportparams( o )
  if o[:segmode]
    #in segmenta mode
    #segmentation_mode(o)
    rnncv(o)
  elseif o[:conmode]
    context_mode(o)
  else
    generation_mode(o)
  end
  println("CHECK SCALEFACTOR")
end
#main()

PROGRAM_FILE == "DrawNet.jl" && main(ARGS)

export revconvertmodel, encode, loaddata, getdata, loaddataset
export getlatentvector, predict, get_mixparams, get_seg_data2
export softmax, sample_gaussian_2d, paddall, appendlatentvec
end
