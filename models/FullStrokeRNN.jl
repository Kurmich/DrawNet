#sequence to sequence variational autoencoder
function s2sVAE(model, data, seqlen; epsilon = 1e-6, istraining::Bool = true, dprob = 0)
  #model settings
  maxlen = maximum(seqlen) #maximum length of the input sequence
  M = Int((size(model[:output][1], 2)-3)/6) #number of mixtures
  (batchsize, V) = size(data[1])
  V = 5
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
    pnorm, mu_x, mu_y, sigma_x, sigma_y, rho, qlognorm = get_mixparams(output, M) #get mixtur parameters and normalized logit values
    mix_probs = pnorm .* vec_bivariate_prob(data[i][:, 1], data[i][:, 2], mu_x, mu_y, sigma_x, sigma_y, rho) #weighted probabilities of mixtures
    mask = 1 .- data[i][:, V] #mask to zero out all terms beyond actual N_s the last actual stroke
    offset_loss += -sum( log( sum(mix_probs, 2).+ epsilon ) .* mask ) #L_s on paper(add epsilon to avoid log(0))
    if istraining
      penstate_loss += -sum(data[i][:, (V-2):V] .* qlognorm) #L_p on paper
    else
      penstate_loss += -sum(data[i][:, (V-2):V] .* qlognorm .* mask) #L_p on paper
    end
  end
  offset_loss /= (maxlen * batchsize)
  penstate_loss /= (maxlen * batchsize)
  kl_loss = -sum((1 + sigma_cap - mu.*mu - exp(sigma_cap))) / (2*z_size*batchsize)   #Kullback-Leibler divergence loss term
  return offset_loss, penstate_loss, kl_loss
end

function ss2vae(model, data, seqlens, wkl, o; istraining = true)
  offset_loss, penstate_loss, kl_loss = 0, 0, 0
  for j = 1:length(data)
    if istraining
      x = perturb(data[j]; scalefactor=o[:scalefactor])
    else
      x = data[j]
    end
    cur_offset_loss, cur_penstate_loss, cur_kl_loss  = s2sVAE(model, map(a->convert(atype, a), x), seqlens[j]; dprob=o[:dprob])
    offset_loss += cur_offset_loss
    penstate_loss += cur_penstate_loss
    kl_loss += cur_kl_loss
  end
  offset_loss /= length(data)
  penstate_loss /= length(data)
  kl_loss /= length(data)
  if istraining
    kl_loss = max(kl_loss, kl_tolerance)
  else
    return penstate_loss, offset_loss, kl_loss
  end
  loss = offset_loss + penstate_loss + wkl*kl_loss
  return loss
end

s2sVAEgrad = grad(ss2vae)
function train(model, trndata, trnseqlens, vlddata, vldseqlens, opts, o)
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      cur_wkl = KL.w - (KL.w - KL.wstart) * ((KL.decayrate)^step)
      cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
      grads = s2sVAEgrad(model, trndata[i], trnseqlens[i], cur_wkl, o)
      updatelr!(opts, cur_lr)
      update!(model, grads, opts)
      step += 1
    end
    (vld_rec_loss, vld_kl_loss) = evaluatemodel(model, vlddata, vldseqlens, KL.w, o)
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

function evaluatemodel(model, data, seqlens, wkl, o)
  rec_loss, kl_loss = 0, 0
  count = 0
  for i = 1:length(data)
    penstate_loss, offset_loss, cur_kl_loss = ss2vae(model,  data[i], seqlens[i], wkl, o; istraining = false)
    rec_loss += (penstate_loss + offset_loss)
    kl_loss += cur_kl_loss
    count += 1
  end
  return rec_loss/count, kl_loss/count
end

function minibatch(sketchpoints3D, numbatches, params)
  #=stroke level minibatching=#
  info("Stroke minibatching")
  data = []
  idm_data = []
  seqlens = []
  for i=0:(numbatches-1)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, params)
    strokeseq, seqlen = getstrokeseqs(x_batch_5D)
    push!(data, strokeseq)
    push!(seqlens, seqlen)
  end
  return data, seqlens
end
