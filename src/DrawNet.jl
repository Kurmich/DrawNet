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
    result = copy(data[i])
    result[:, 1] *= x_scalefactor
    result[:, 2] *= y_scalefactor
    #perturb idm if needed
    push!(pdata, result)
  end
  return pdata
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


function classify_w_context(model, data, f_data, ygold, seqlen, wkl, o; istraining::Bool = true, dprob = 0, weights=nothing)
  maxlen = length(data) #maximum length of the input sequence
  (batchsize, V) = size(data[1])
  d_H = size(model[:pred][1], 1) #decoder hidden unit size
  z_size = size(model[:sigma_cap][1], 2) #size of latent vector z
  #println(size(f_data[1]), size(data[1]), size(model[:fw_embed][1]))
  h = encode(model, f_data, length(f_data), batchsize; dprob=dprob, attn=o[:attn]) #encode using full sketch
#  println("length f_data ", length(f_data), " size f_data[1] ", size(f_data[1]))
  #predecoder step
  mu = h*model[:mu][1] .+ model[:mu][2]
  sigma_cap = h*model[:sigma_cap][1] .+ model[:sigma_cap][2]
  sigma = exp( sigma_cap/2 )
  z = mu + sigma .* atype( gaussian(batchsize, z_size; mean=0.0, std=1.0) )

#  println("size z", size(z))
  #println("size data[1]", size(data[1]))
  #decoder step
  hc = tanh(z * model[:z][1] .+ model[:z][2])
  state = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
  w = AutoGrad.getval(model)
  hasembed, hasshift = haskey(w, :embed), haskey(w, :dec_shifts)
  alpha, beta  = hasshift ? (model[:dec_shifts][1], model[:dec_shifts][2]) : (nothing, nothing)
  #println("classify_w_context maxlen ", maxlen)
  for i = 1:maxlen
    input = hcat(data[i], z)
    #println("knet arr ", Array(AutoGrad.getval(input)))
    if hasembed
      input = input * model[:embed]
    end
    #mask = 1 .- input[:, V]
    state = lstm(model[:decode], state, input; alpha=alpha, beta=beta, dprob=dprob)
  end
  ypred = state[1]*model[:pred][1] .+ model[:pred][2]
  ynorm = logp(ypred, 2)
  softloss = -sum(ygold .* ynorm .* weights)/batchsize
  kl_loss = -sum((1 + sigma_cap - mu.*mu - exp(sigma_cap))) / (2*z_size*batchsize)   #Kullback-Leibler divergence loss term

  if istraining
    kl_loss = max(kl_loss, kl_tolerance)
  else
    return softloss, kl_loss, ypred
  end
  loss = softloss + wkl*kl_loss
  return loss
end



function eval_w_context(model, data, f_data, seqlens, ygold, o; genmodel = nothing, weights = nothing)
  correct_count = zeros(1, size(ygold[1], 2))
  instance_count = zeros(1, size(ygold[1], 2))
  curloss, curright = 0.0, 0.0
  loss, kloss, correct = 0.0, 0.0, 0.0
  ygold = map(a->convert(atype, a), ygold)
  count = 0.0
  for i = 1:length(data)
    for j = 1:length(data[i])
      softloss, kl_loss, ypred = classify_w_context(model, map(a->convert(atype, a), data[i][j]), map(a->convert(atype, a), f_data[i]), ygold[i], seqlens[i][j], 0, o; istraining=false, dprob=0, weights=weights)
      correct_count, instance_count = countcorrect(Array(ypred), Array(ygold[i]), correct_count, instance_count)
      loss += softloss
      kloss += kl_loss
      count += size(ygold[i], 1) #CHECK
    end
  end
  return loss/count, kloss/count, correct_count, instance_count
end

gradcon = grad(classify_w_context)
function segment_w_context(model, dataset, opts, o; genmodel = nothing)
  (trndata, trnseqlens, trngold, trnstats, f_trndata, f_trnseqlens)  = dataset[:trn]
  (vlddata, vldseqlens, vldgold, vldstats, f_vlddata, f_vldseqlens) = dataset[:vld]
  (tstdata, tstseqlens, tstgold, tststats, f_tstdata, f_tstseqlens) = dataset[:tst]
  append!(trndata, vlddata)
  append!(trnseqlens, vldseqlens)
  append!(f_trndata, f_vlddata)
  append!(f_trnseqlens, f_vldseqlens)
  append!(trngold, vldgold)
  trnstats += vldstats
  (vlddata, vldseqlens, vldgold, vldstats) = (tstdata, tstseqlens, tstgold, tststats)
  f_vlddata = f_tstdata
  println(trnstats)
  println(-trnstats/maximum(trnstats))
  weights = softmax(-trnstats/maximum(trnstats), 2) # per class weights for loss function
  #weights[5] *= 0.3
  #weights = ones(size(weights))
  println(weights)
  flush(STDOUT)
  weights = atype(weights)
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  trngold = map(a->convert(atype, a), trngold)
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      for j = 1:length(trndata[i])
        @assert(length(trndata[i]) == 1)
        cur_wkl = KL.w - (KL.w - KL.wstart) * ((KL.decayrate)^step)
        cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
        x = perturb(trndata[i][j]; scalefactor=o[:scalefactor])
        #println("input arr ", x[2])
        #println(size(f_trndata[i][1]))
        f_x = perturb(f_trndata[i]; scalefactor=o[:scalefactor])
        #println(size(f_x[1]))
        grads = gradcon(model, map(a->convert(atype, a), x), map(a->convert(atype, a), f_x), trngold[i], trnseqlens[i][j], cur_wkl, o; dprob=o[:dprob], weights=weights)
        updatelr!(opts, cur_lr)
        update!(model, grads, opts)
        step += 1
        #vld_soft_loss, vld_kl_loss, correct_count, instance_count = eval_w_context(model, vlddata, f_vlddata, vldseqlens, vldgold, o; genmodel=genmodel, weights=weights)
        #@printf("vld data - epoch: %d step: %d lr: %g vld softloss loss: %g vld kl loss: %g total acc: %g\n", e, step, cur_lr, vld_soft_loss, vld_kl_loss, sum(correct_count)/sum(instance_count))
      end
    end
    vld_soft_loss, vld_kl_loss, correct_count, instance_count = eval_w_context(model, vlddata, f_vlddata, vldseqlens, vldgold, o; genmodel=genmodel, weights=weights)
    vld_loss = vld_soft_loss + vld_kl_loss
    #save the best model
    if vld_loss < best_vld_cost
      best_vld_cost = vld_loss
      arrmodel = convertmodel(model)
      println("Epoch: $(e) saving best model to $(pretrnp)$(o[:bestmodel])")
      save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
    end
    #report losses
    @printf("vld data - epoch: %d step: %d lr: %g vld softloss loss: %g vld kl loss: %g total acc: %g\n", e, step, cur_lr, vld_soft_loss, vld_kl_loss, sum(correct_count)/sum(instance_count))
    for c = 1:length(correct_count)
      @printf("vld data - epoch: %d class %d instances: %d acc: %g \n", e, c, instance_count[c], correct_count[c]/instance_count[c] )
    end
    #save every o[:save_every] epochs
    if e%o[:save_every] == 0
      arrmodel = convertmodel(model)
      save("$(pretrnp)m$(e)$(o[:tmpmodel])","model", arrmodel)
    end
    flush(STDOUT)
  end
end

function transclassify_full(model, genmodel, data, seqlen, ygold, o; istraining::Bool = true, weights = nothing, fullgenmodel=nothing, fulldata = nothing)
  seqlen = length(data)
  batchsize = size(data[1], 1)
  if !istraining
    h = encode(genmodel, data, seqlen, batchsize; dprob=0)
    if fullgenmodel != nothing
      h_full = encode(fullgenmodel, fulldata, length(fulldata), batchsize; dprob=0)
      h = hcat(h, h_full)
    end
    h = h*model[:w1][1] .+ model[:w1][2]
    ypred = h*model[:pred][1] .+ model[:pred][2]
    ynorm = logp(ypred, 2)
    return -sum(ygold .* ynorm)/size(ygold, 1), ypred
  end
  h = encode(genmodel, data, seqlen, batchsize; dprob=o[:dprob])
  if fullgenmodel != nothing
    h_full = encode(fullgenmodel, fulldata, length(fulldata), batchsize; dprob=o[:dprob])
    h = hcat(h, h_full)
  end
  h = h*model[:w1][1] .+ model[:w1][2]
  ypred = h*model[:pred][1] .+ model[:pred][2]
  ynorm = logp(ypred, 2)
  if weights == nothing
    return -sum(ygold .* ynorm)/size(ygold, 1)
  end
  return -sum(ygold .* ynorm .* weights)/size(ygold, 1)
end



function classify(model, data, seqlen, ygold, o; epsilon = 1e-6, istraining::Bool = true, weights = nothing, istate = nothing , z = nothing)
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

function evalsegm_full(model, data, f_data, seqlens, ygold, o; genmodel = nothing, fullgenmodel = nothing, weights = nothing)
  correct_count = zeros(1, size(ygold[1], 2))
  instance_count = zeros(1, size(ygold[1], 2))
  curloss, curright = 0.0, 0.0
  loss, correct = 0.0, 0.0
  ygold = map(a->convert(atype, a), ygold)
  count = 0.0
  for i = 1:length(data)
    for j = 1:length(data[i])
      if o[:hascontext]
        #d_H = size(genmodel[:output][1], 1)
        x = map(a->convert(atype, a), data[i][j])
        #z = getlatentvector(genmodel, x)
        #=x = map(a->convert(atype, a), data[i][j])
        f_x = map(a->convert(atype, a), f_data[i])
        z = getlatentvector(genmodel, f_x)
        x = appendlatentvec(x, z)=#
        #curloss, ypred = classify(model, x, seqlens[i][j], ygold[i], o; istraining=false, z = z, weights=weights)
        f_x = map(a->convert(atype, a), f_data[i])
        curloss, ypred = transclassify(model, genmodel, x, seqlens[i][j], ygold[i], o; istraining = false, fullgenmodel= fullgenmodel, weights=weights, fulldata=f_x)
      else
        curloss, ypred = classify(model, map(a->convert(atype, a), data[i][j]), seqlens[i][j], ygold[i], o; istraining=false, weights=weights)
      end
      correct_count, instance_count = countcorrect(Array(ypred), Array(ygold[i]), correct_count, instance_count)
      loss += curloss
      count += size(ygold[i], 1) #CHECK
    end
  end
  return loss/count, correct_count, instance_count
end



function transclassify(model, genmodel, data, seqlen, ygold, o; istraining::Bool = true, weights = nothing)
  seqlen = length(data)
  batchsize = size(data[1], 1)
  if !istraining
    h = encode(genmodel, data, seqlen, batchsize; dprob=0)
    h = h*model[:w1][1] .+ model[:w1][2]
    ypred = h*model[:pred][1] .+ model[:pred][2]
    ynorm = logp(ypred, 2)
    return -sum(ygold .* ynorm)/size(ygold, 1), ypred
  end
  h = encode(genmodel, data, seqlen, batchsize; dprob=o[:dprob])
  h = h*model[:w1][1] .+ model[:w1][2]
  ypred = h*model[:pred][1] .+ model[:pred][2]
  ynorm = logp(ypred, 2)
  if weights == nothing
    return -sum(ygold .* ynorm)/size(ygold, 1)
  end
  return -sum(ygold .* ynorm .* weights)/size(ygold, 1)
end


function evalsegm(model, data, f_data, seqlens, ygold, o; genmodel = nothing, weights = nothing)
  correct_count = zeros(1, size(ygold[1], 2))
  instance_count = zeros(1, size(ygold[1], 2))
  curloss, curright = 0.0, 0.0
  loss, correct = 0.0, 0.0
  ygold = map(a->convert(atype, a), ygold)
  count = 0.0
  for i = 1:length(data)
    for j = 1:length(data[i])
      if o[:hascontext]
        #d_H = size(genmodel[:output][1], 1)
        x = map(a->convert(atype, a), data[i][j])
        #z = getlatentvector(genmodel, x)
        #=x = map(a->convert(atype, a), data[i][j])
        f_x = map(a->convert(atype, a), f_data[i])
        z = getlatentvector(genmodel, f_x)
        x = appendlatentvec(x, z)=#
        #curloss, ypred = classify(model, x, seqlens[i][j], ygold[i], o; istraining=false, z = z, weights=weights)
        #f_x = map(a->convert(atype, a), f_data[i])
        curloss, ypred = transclassify(model, genmodel, x, seqlens[i][j], ygold[i], o; istraining = false, weights=weights)
      else
        curloss, ypred = classify(model, map(a->convert(atype, a), data[i][j]), seqlens[i][j], ygold[i], o; istraining=false, weights=weights)
      end
      correct_count, instance_count = countcorrect(Array(ypred), Array(ygold[i]), correct_count, instance_count)
      loss += curloss
      count += size(ygold[i], 1) #CHECK
    end
  end
  return loss/count, correct_count, instance_count
end

gradtrans = grad(transclassify)
gradloss = grad(classify)
function segment(model, dataset, opts, o; genmodel = nothing, fullgenmodel = nothing)
  (trndata, trnseqlens, trngold, trnstats, f_trndata, f_trnseqlens)  = dataset[:trn]
  (vlddata, vldseqlens, vldgold, vldstats, f_vlddata, f_vldseqlens) = dataset[:vld]
  (tstdata, tstseqlens, tstgold, tststats, f_tstdata, f_tstseqlens) = dataset[:tst]
  append!(trndata, vlddata)
  append!(trnseqlens, vldseqlens)
  append!(trngold, vldgold)
  append!(trndata, tstdata)
  append!(trnseqlens, tstseqlens)
  append!(trngold, tstgold)
  append!(f_trndata, f_vlddata)
  append!(f_trnseqlens, f_vldseqlens)

  trnstats += vldstats
  (vlddata, vldseqlens, vldgold, vldstats) = (trndata, trnseqlens, trngold, trnstats)
  f_vlddata = f_tstdata
  println(trnstats)
  println(-trnstats/maximum(trnstats))
  weights = softmax(-trnstats/maximum(trnstats), 2) # per class weights for loss function
  println(weights)
  flush(STDOUT)
  weights = atype(weights)
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  trngold = map(a->convert(atype, a), trngold)
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      for j = 1:length(trndata[i])
        @assert(length(trndata[i]) == 1)
        cur_wkl = KL.w - (KL.w - KL.wstart) * ((KL.decayrate)^step)
        cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
        x = perturb(trndata[i][j]; scalefactor=o[:scalefactor])
        if o[:hascontext]
          #d_H = size(genmodel[:output][1], 1)
          #x = map(a->convert(atype, a), x)
          #z = getlatentvector(genmodel, x)
          #=x = map(a->convert(atype, a), x)
          f_x = map(a->convert(atype, a), f_trndata[i])
          z = getlatentvector(genmodel, f_x)
          x = appendlatentvec(x, z)=#
          #grads = gradloss(model, x, trnseqlens[i][j], trngold[i], o; weights=weights, z=z)
          #f_x = map(a->convert(atype, a), f_trndata[i])
          x = map(a->convert(atype, a), x)
          grads = gradtrans(model, genmodel, x, trnseqlens[i][j], trngold[i], o; istraining = true, weights=weights)
        else
          grads = gradloss(model, map(a->convert(atype, a), x), trnseqlens[i][j], trngold[i], o; weights=weights)
        end
        updatelr!(opts, cur_lr)
        update!(model, grads, opts)
        step += 1
        #=vld_loss, correct_count, instance_count = evalsegm(model, vlddata, vldseqlens, vldgold, o)
        #save the best model
        if vld_loss < best_vld_cost
          best_vld_cost = vld_loss
          arrmodel = convertmodel(model)
          println("Step: $(step) saving best model to $(pretrnp)$(o[:bestmodel])")
          save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
          @printf("vld data - epoch: %d step: %d lr: %g vld loss: %g total acc: %g\n", e, step, cur_lr, vld_loss, sum(correct_count)/sum(instance_count))
        end=#
      end
    end
    vld_loss, correct_count, instance_count = evalsegm(model, vlddata, f_vlddata, vldseqlens, vldgold, o; genmodel=genmodel, weights=weights)
    #save the best model
    if vld_loss < best_vld_cost
      best_vld_cost = vld_loss
      arrmodel = convertmodel(model)
      println("Epoch: $(e) saving best model to $(pretrnp)$(o[:bestmodel])")
      save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
    end
    #report losses
    @printf("vld data - epoch: %d step: %d lr: %g wkl vld loss: %g total acc: %g\n", e, step, cur_lr, vld_loss, sum(correct_count)/sum(instance_count))
    for c = 1:length(correct_count)
      @printf("vld data - epoch: %d; class %d; instances: %d; correct instances: %d; acc: %g \n", e, c, instance_count[c], correct_count[c], correct_count[c]/instance_count[c] )
    end
    #save every o[:save_every] epochs
    if e%o[:save_every] == 0
      arrmodel = convertmodel(model)
      save("$(pretrnp)m$(e)$(o[:tmpmodel])","model", arrmodel)
    end
    flush(STDOUT)
  end
end


function getdata(filename, params)
  return getsketchpoints3D(filename; params = params)
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

function makebatches(sketches, points3D, f_points3D, labels, params; full::Bool = false)
  #=creates batches of annotated data=#
  batch_count = Int(ceil(length(points3D)/params.batchsize))
  params.numbatches = batch_count
  f_data, f_seqlens =  sketch_minibatch(f_points3D, batch_count, params)
  data, seqlens = minibatch(points3D, batch_count, params)
  ygolds, instance_per_label = getlabels(sketches, labels, batch_count, params)
  @assert(length(ygolds) == length(data))
  return (data, seqlens, ygolds, instance_per_label, f_data, f_seqlens)
end

function makedataset(trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, f_trnpoints3D, f_vldpoints3D, f_tstpoints3D, labels, params)
  dataset = Dict()
  dataset[:trn] = makebatches(trnsketches, trnpoints3D, f_trnpoints3D, labels, params)
  dataset[:vld] = makebatches(vldsketches, vldpoints3D, f_vldpoints3D, labels, params)
  dataset[:tst] = makebatches(tstsketches, tstpoints3D, f_tstpoints3D, labels, params)
  return dataset
end


function make_gen_dataset(trnpoints3D, vldpoints3D, tstpoints3D, params)
  #minibatch
  trn_batch_count = div(length(trnpoints3D), params.batchsize)
  params.numbatches = trn_batch_count
  trndata, trnseqlens = minibatch(trnpoints3D, trn_batch_count-1, params)
  vld_batch_count = div(length(vldpoints3D), params.batchsize)
  params.numbatches = vld_batch_count
  vlddata, vldseqlens= minibatch(vldpoints3D, vld_batch_count-1, params)
  tst_batch_count = div(length(tstpoints3D), params.batchsize)
  params.numbatches = tst_batch_count
  tstdata, tstseqlens = minibatch(tstpoints3D, tst_batch_count-1, params)
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

  #get sketches
  sketchpoints3D, numbatches, sketches = getdata(o[:filename], params)
  trnidx, vldidx, tstidx = splitdata(sketchpoints3D)

  info("data was split")
  trnpoints3D, vldpoints3D, tstpoints3D = get_splitteddata(sketchpoints3D, trnidx, vldidx, tstidx)
  trnsketches, vldsketches, tstsketches = get_splitteddata(sketches, trnidx, vldidx, tstidx)
  #normalize sketches
  info("In nomralization phase")
  normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params)
  #save the data
  savedata("idx$(o[:filename])", trnidx, vldidx, tstidx)
  savedata("data$(o[:filename])", trnpoints3D, vldpoints3D, tstpoints3D)

  dataset = make_gen_dataset(trnpoints3D, vldpoints3D, tstpoints3D, params)
  return dataset
end

function makesplit(sketches, labels, vldsize; trn_tst_indices = nothing, trn_vld_indices = nothing, params = nothing)
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
  trndata = shuffle(dict2list(trndict))  #as list ,> this is list of lists we need just list of sketches
  vlddata = shuffle(dict2list(vlddict))
  tstdata = shuffle(dict2list(tstdict)) #as list
  @printf("trnsize: %d vldsize: %d tstsize: %d \n", length(trndata), length(vlddata), length(tstdata))
  trnpoints3D, numbatches, trnsketches = preprocess(trndata, params)
  vldpoints3D, numbatches, vldsketches = preprocess(vlddata, params)
  tstpoints3D, numbatches, tstsketches = preprocess(tstdata, params)
  println("IN NORMALIZATION PHASE")
  normalizedata!(trnpoints3D, vldpoints3D, tstpoints3D, params; scalefactor=43.8)
  @printf("3D trnsize: %d vldsize: %d tstsize: %d \n", length(trnpoints3D), length(vldpoints3D), length(tstpoints3D))
  return trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, trn_tst_indices, trn_vld_indices
end

function get_seg_data(filename, labels, vldsize; trn_tst_indices = nothing, trn_vld_indices = nothing, params = nothing)
  #=Create dataset for segmentation model=#
  filename = string(annotp, filename)
  #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]

  annotations = getannotateddata(filename, labels)
  sketches, full_sketches = annotated2sketch_obj(annotations)
  for (key, value) in sketches
      println(key, " ==> ", length(value))
  end
  trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, trn_tst_indices, trn_vld_indices = makesplit(sketches, labels, vldsize; trn_tst_indices = trn_tst_indices, trn_vld_indices = trn_vld_indices, params = params)
  f_trnsketches, f_trnpoints3D, f_vldsketches, f_vldpoints3D, f_tstsketches, f_tstpoints3D, f_trn_tst_indices, f_trn_vld_indices = makesplit(full_sketches, labels, vldsize; trn_tst_indices = trn_tst_indices, trn_vld_indices = trn_vld_indices, params = params)
  dataset = makedataset(trnsketches, trnpoints3D, vldsketches, vldpoints3D, tstsketches, tstpoints3D, f_trnpoints3D, f_vldpoints3D, f_tstpoints3D, labels, params)
  @printf("# of trn sketches: %d  # of trn batches: %d  \n", length(trnpoints3D), length(f_trnsketches))
  @printf("# of vld sketches: %d  # of vld batches: %d \n", length(vldpoints3D), length(f_vldpoints3D))
  return dataset, trn_tst_indices, trn_vld_indices
end

function rnncv(o)
  #=RNN cross validation=#
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  #labels = [ "L", "F", "FP"]
  labels = ["W", "B", "T", "WNDW", "FA"]
  o[:numclasses] = length(labels)
  model = initransfer(o) #initsegmenter(o)
  params = Parameters()
  params.batchsize = o[:batchsize]
  params.min_seq_length = 1
  global optim = initoptim(model, o[:optimization])
  vldsize = 1 / o[:cvfolds]
  smooth = true
  rawname = split(o[:filename], ".")[1]
  if !o[:readydata]
    dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; params=params)
  else
    #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
    trn_tst_indices = load("$(rawname)trn_tst_indices.jld")["indices"]
    trn_vld_indices = load("$(rawname)trn_vld_indices.jld")["indices"]
    #dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; trn_tst_indices=trn_tst_indices, trn_vld_indices=trn_vld_indices, params=params)
  end
  println("Starting training")
  flush(STDOUT)
  if o[:hascontext]
    println("Loading context model.")
    w = load("$(pretrnp)$(o[:model])")
    genmodel = revconvertmodel(w["model"])
    println("Loading full context model.")
    w = load("$(pretrnp)$(o[:fullmodel])")
    fullgenmodel = revconvertmodel(w["model"])
  else
    genmodel = nothing
    fullgenmodel = nothing
  end
  if o[:fold] == -1
    for i=1:o[:cvfolds]
      params.max_seq_length = 100 #SOLVE THIS ISSUE
      println("Fold $(i) is Starting")
      segment(model, dataset, optim, o; genmodel=genmodel)
      shift_indices!(trn_tst_indices, vldsize)
      model = initsegmenter(o)
      optim = initoptim(model, o[:optimization])
      dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; trn_tst_indices=trn_tst_indices, params=params)
    end
  else
    println("Fold $(o[:fold])")
    for i=1:(o[:fold]-1)
      println("Shifting $(i)")
      shift_indices!(trn_tst_indices, vldsize)
    end
    params.max_seq_length = 100
    dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; trn_tst_indices=trn_tst_indices, trn_vld_indices=trn_vld_indices, params=params)
    segment(model, dataset, optim, o; genmodel=genmodel, fullgenmodel=fullgenmodel)
  end
end


function segmentation_mode(o)
  global const KL = KLparameters(o[:wkl], o[:kl_weight_start], o[:kl_decay_rate]) #Kullback-Leibler(KL) parameters
  global const LRP = LRparameters(o[:lr], o[:minlr], o[:lr_decay_rate]) #Learning Rate Parameters(LRP)
  global const kl_tolerance = o[:kl_tolerance]
  #labels = [ "L", "F", "FP"]
  labels = ["W", "B", "T", "WNDW", "FA"]
  o[:numclasses] = length(labels)
  model = init_con_segmenter(o)
  params = Parameters()
  params.batchsize = o[:batchsize]
  params.min_seq_length = 1
  global optim = initoptim(model, o[:optimization])
  vldsize = 1/5
  smooth = true
  rawname = split(o[:filename], ".")[1]
  if !o[:readydata]
    dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; params=params)
  else
    #labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
    trn_tst_indices = load("$(rawname)trn_tst_indices.jld")["indices"]
    trn_vld_indices = load("$(rawname)trn_vld_indices.jld")["indices"]
    dataset, trn_tst_indices, trn_vld_indices = get_seg_data(o[:filename], labels, vldsize; trn_tst_indices=trn_tst_indices, trn_vld_indices=trn_vld_indices, params=params)
  end

  println("Starting training")
  flush(STDOUT)
  if o[:hascontext]
    println("Loading context model.")
    w = load("$(pretrnp)$(o[:model])")
    genmodel = revconvertmodel(w["model"])
    w = load("$(pretrnp)$(o[:fullmodel])")
    fullgenmodel = revconvertmodel(w["model"])
  else
    genmodel = nothing
  end
  segment_w_context(model, dataset, optim, o; genmodel=genmodel, fullgenmodel=fullgenmodel)
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
    trnpoints3D, vldpoints3D, tstpoints3D = loaddata("$(datap)data$(o[:dataset])")
    dataset = make_gen_dataset(trnpoints3D, vldpoints3D, tstpoints3D, params)
  #  trnidm, vldidm, tstidm  = loaddata("$(datap)idm$(o[:imlen])$(o[:dataset])")
  end
  println("Starting training")
  reportmodel(model)
#  trndata = paddall(trndata, trnidmtuples, o[:imlen])
  #vlddata = paddall(vlddata, vldidmtuples, o[:imlen])
  #info("padding was complete")
  flush(STDOUT)
  train(model, dataset, optim, o)
end


function reportparams( o )
  println("Has Attention: $(o[:attn]); Mean Representation: $(o[:meanrep]); GMM Context: $(o[:hascontext])")
  println("Data filename: $(o[:filename])")
  println("Optimizer: $(o[:optimization])")
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
    ("--sfilename"; arg_type=String; default="airplane.ndjson"; help="Data file name")
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
    ("--optimization"; default="Adam(;gclip = 1.0)"; help="Optimization algorithm and parameters.")
    ("--dataset"; arg_type=String; default="full_simplified_airplane.jld"; help="Name of the dataset")
    ("--fullmodel"; arg_type=String; default="best_gen_full_airmax50min5.jld"; help="Name of the pretrained model")
    ("--model"; arg_type=String; default="model100.jld"; help="Name of the pretrained model")
    ("--cvfolds"; arg_type=Int; default=5; help="Number of folds to use for cross validation.")
    ("--fold"; arg_type=Int; default=-1; help="Current fold.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  reportparams( o )
  if o[:segmode]
    #in segmenta mode
    #segmentation_mode(o)
    rnncv(o)
  else
    generation_mode(o)
  end
end
#main()
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "DrawNet.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

export revconvertmodel, encode, loaddata, getdata
export getlatentvector, predict, get_mixparams
export softmax, sample_gaussian_2d, paddall, appendlatentvec
end
