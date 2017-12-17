
#sequence to sequence variational autoencoder
function s2sVAE(model, data, seqlen, wkl; epsilon = 1e-6, istraining::Bool = true, dprob = 0)
  #model settings
  maxlen = maximum(seqlen) #maximum length of the input sequence
  (batchsize, V) = size(data[1])
  M = Int((size(model[:output][1], 2)-(V-2))/6) #number of mixtures
  #V = 5
  #println(V)
  d_H = size(model[:output][1], 1) #decoder hidden unit size
  z_size = size(model[:sigma_cap][1], 2) #size of latent vector z
  h = encode(model, data, maxlen, batchsize; dprob=dprob)
#  h = hcat(h, avgim)
  #predecoder step
  mu = h*model[:mu][1] .+ model[:mu][2]
  sigma_cap = h*model[:sigma_cap][1] .+ model[:sigma_cap][2]
  sigma = exp( sigma_cap/2 )
  z = mu + sigma .* atype( gaussian(batchsize, z_size; mean=0.0, std=1.0) )

  #decoder step
  hc = tanh(z * model[:z][1] .+ model[:z][2])
  state = (hc[:, 1:d_H], hc[:, d_H+1:2d_H])
  penstate_loss = 0
  offset_loss = 0
  w = AutoGrad.getval(model)
  hasembed, hasshift = haskey(w, :embed), haskey(w, :dec_shifts)
  alpha, beta  = hasshift ? (model[:dec_shifts][1], model[:dec_shifts][2]) : (nothing, nothing)
  for i = 2:maxlen
    #dims data = [batchsize, V] = [batchsize, 5]
    input = hcat(data[i-1][:, 1:V], z) #concatenate latent vector with previous point
    if hasembed
      input = input * model[:embed]
    end
    state = lstm(model[:decode], state, input; alpha=alpha, beta=beta, dprob=dprob)
    output =  predict(model[:output], state[1]) #get output params
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qlognorm = get_mixparams(output, M, V) #get mixtur parameters and normalized logit values
    mix_probs = pnorm .* vec_bivariate_prob(data[i][:, 1], data[i][:, 2], mu_x, mu_y, sigma_x, sigma_y, rho) #weighted probabilities of mixtures
    mask = 1 .- data[i][:, V] #mask to zero out all terms beyond actual N_s the last actual stroke
    offset_loss += -sum( log( sum(mix_probs, 2).+ epsilon ) .* mask ) #L_s on paper(add epsilon to avoid log(0))
    if istraining
      penstate_loss += -sum(data[i][:, 3:V] .* qlognorm) #L_p on paper
    else
      penstate_loss += -sum(data[i][:, 3:V] .* qlognorm .* mask) #L_p on paper
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
  #println("$(AutoGrad.getval(offset_loss)) $(AutoGrad.getval(penstate_loss)) $(AutoGrad.getval(kl_loss))")
  loss = offset_loss + penstate_loss + wkl*kl_loss
  return loss
end


s2sVAEgrad = grad(s2sVAE)
function train(model, dataset, opts, o)
  (trndata, trnseqlens) = dataset[:trn]
  (vlddata, vldseqlens) = dataset[:vld]
  (tstdata, tstseqlens) = dataset[:tst]
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      cur_wkl = KL.w - (KL.w - KL.wstart) * ((KL.decayrate)^step)
      cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
      x = perturb(trndata[i]; scalefactor=o[:scalefactor])
      grads = s2sVAEgrad(model, map(a->convert(atype, a), x), trnseqlens[i], cur_wkl; dprob=o[:dprob])
      updatelr!(opts, cur_lr)
      update!(model, grads, opts)
      step += 1
    end
    (vld_rec_loss, vld_kl_loss) = evaluatemodel(model, vlddata, vldseqlens, KL.w)
    #save the best model
    vld_cost = vld_rec_loss + KL.w * vld_kl_loss
    if vld_cost < best_vld_cost
      best_vld_cost = vld_cost
      arrmodel = convertmodel(model)
      println("Epoch: $(e) saving best model to $(pretrnp)$(o[:bestmodel])")
      save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
    end
    #report losses
    @printf("vld data - epoch: %d step %d rec loss: %g KL loss: %g  wkl: %g lr: %g \n", e, step, vld_rec_loss, vld_kl_loss, cur_wkl, cur_lr)
    #save every o[:save_every] epochs
    if e%o[:save_every] == 0
      arrmodel = convertmodel(model)
      save("$(pretrnp)m$(e)$(o[:tmpmodel])","model", arrmodel)
    end
    flush(STDOUT)
  end
end

function evaluatemodel(model, data, seqlens, wkl)
  rec_loss, kl_loss = 0, 0
  count = 0
  for i = 1:length(data)
    penstate_loss, offset_loss, cur_kl_loss = s2sVAE(model, map(a->convert(atype, a), data[i]), seqlens[i], wkl; istraining = false)
    rec_loss += (penstate_loss + offset_loss)
    kl_loss += cur_kl_loss
    count += 1
  end
  return rec_loss/count, kl_loss/count
end


function getlabels(sketches, labels, params)
  batch_count = div(length(sketches), params.batchsize)
  params.numbatches = batch_count

  numlabels = length(labels) #number of classes
  instance_per_label = zeros(1, numlabels)
  onehotvecs = []
  @assert(batch_count > 0)
  for idx=0:(batch_count-1)
    start_ind = idx * params.batchsize
    end_ind = min( (start_ind + params.batchsize), length(sketches))
    indices = (start_ind + 1) : end_ind
    ygolds = zeros(end_ind-start_ind, numlabels)
    batch = sketches[indices]
    for i=1:length(batch)
      sketch = batch[i]
      class = findfirst(labels, sketch.label)
      ygolds[i, class] = 1
      instance_per_label[class] += 1
    end
    push!(onehotvecs, ygolds)
  end
  return onehotvecs, instance_per_label
end

function get_avg_idms(sketches, f_sketches, params)
  batch_count = div(length(sketches), params.batchsize)
  params.numbatches = batch_count
  avg_idms = []
  for idx=0:(batch_count-1)
    start_ind = idx * params.batchsize
    end_ind = min( (start_ind + params.batchsize), length(sketches))
    indices = (start_ind + 1) : end_ind
    batch = f_sketches[indices]
    strokebatch = sketches[indices]
    idms = nothing
    for i=1:length(batch)
      sketch = batch[i]
      stroke = strokebatch[i]
      #get mid and end points of the stroke
      mid = (sum(stroke.points, 2)/(size(stroke.points, 2)*256))'
      spnt =  stroke.points[:, 1]'/256
      epnt = stroke.points[:, size(stroke.points,2)]'/256
      #get average idm of full sketch
      fullidm = extractidm(sketch.points, sketch.end_indices)
      #println(size(spnt), size(fullidm))
      fullidm = hcat(fullidm, mid)
      fullidm = hcat(fullidm, spnt)
      fullidm = hcat(fullidm, epnt)
      if idms == nothing
        idms = fullidm
      else
        idms = vcat(idms, fullidm)
      end
    end
    push!(avg_idms, idms)
  end
  return avg_idms
end

function getsketchbatch(x_batch_5D)
  info("Minibatching")
  data = []
  seqlens = []
  sequence = []
  for j=1:size(x_batch_5D, 2)
    points = x_batch_5D[:, j, :]'
    push!(sequence, points)
  end
  push!(data, sequence)
  push!(seqlens, seqlen)
end

function sketch_minibatch(sketchpoints3D, V, params)
  info("Sketch minibatching")
  batch_count = div(length(sketchpoints3D), params.batchsize)
  params.numbatches = batch_count
  data = []
  #idm_data = []
  seqlens = []
  V = 5
  for i=0:(batch_count-1)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, V, params)
    #idm_avg_batch, idm_stroke_batch = get_idm_batch(idmtuples, i, params)
    sequence = []
    for j=1:size(x_batch_5D, 2)
      points = x_batch_5D[:, j, :]'
      push!(sequence, points)
    end
    push!(data, sequence)
    push!(seqlens, seqlen)
    #push!(idm_data, (idm_avg_batch, idm_stroke_batch))
  end
  return data, seqlens#, idm_data
end


function minibatch(sketchpoints3D, V, params)
  #=stroke level minibatching=#
  info("Stroke minibatching")
  batch_count = div(length(sketchpoints3D), params.batchsize)
  params.numbatches = batch_count
  stroke_batches = []
  stroke_seqlens = []
  for i=0:(batch_count-1)
    #x_batch, x_batch_4D, seqlen = getbatch(sketchpoints3D, i, V, params)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, V, params)
    strokeseq, seqlen = getstrokeseqs(x_batch_5D)
    append!(stroke_batches, strokeseq)
    append!(stroke_seqlens, seqlen)
  end
  return stroke_batches, stroke_seqlens
end

function minibatch4d(sketchpoints3D, V, params)
  #=stroke level minibatching=#
  info("Stroke minibatching")
  batch_count = div(length(sketchpoints3D), params.batchsize)
  params.numbatches = batch_count
  stroke_batches = []
  stroke_seqlens = []
  for i=0:(batch_count-1)
    #x_batch, x_batch_4D, seqlen = getbatch(sketchpoints3D, i, V, params)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, V, params)
    strokeseq, seqlen = getstrokeseqs4d(x_batch_5D)
    append!(stroke_batches, strokeseq)
    append!(stroke_seqlens, seqlen)
  end
  return stroke_batches, stroke_seqlens
end
