
function bi_lstm_gen(model, data, seqlens; epsilon = 1e-6, istraining::Bool = true, dprob = 0)
  #Initialize states for forward-backward rnns
  maxlen = length(data) #maximum length of the input sequence
  (batchsize, V) = size(data[1])
  M = Int((size(model[:fw_output][1], 2)-3)/6)
  penstate_loss, offset_loss = 0, 0
  statefw = initstate(batchsize, model[:fw_state0])
  statebw = initstate(batchsize, model[:bw_state0])
  #forward encoder
  w = AutoGrad.getval(model)
  hasembed, hasshift = haskey(w, :fw_embed), haskey(w, :fw_shifts)
  alpha, beta  = hasshift ? (model[:fw_shifts][1], model[:fw_shifts][2]) : (nothing, nothing)
  for i = 1:maxlen
    #println(size(data[i]), size(model[:fw_embed]))
    input = hasembed ? data[i] * model[:fw_embed] : data[i]
    statefw = lstm(model[:fw_encode], statefw, input; alpha=alpha, beta=beta, dprob=dprob)

    output =  predict(model[:fw_output], statefw[1])
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
  #backward encoder
  hasembed, hasshift = haskey(w, :bw_embed), haskey(w, :bw_shifts)
  alpha, beta  = hasshift ? (model[:bw_shifts][1], model[:bw_shifts][2]) : (nothing, nothing)
  for i = maxlen:-1:1
    input = hasembed ? data[i]*model[:bw_embed] : data[i]
    statebw = lstm(model[:bw_encode], statebw, input; alpha=alpha, beta=beta, dprob=dprob)

    output =  predict(model[:bw_output], statebw[1])
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
  if !istraining
    return penstate_loss, offset_loss
  end
  return penstate_loss + offset_loss
end

contextgrad = grad(bi_lstm_gen)
function contrain(model, dataset, opts, o)
  (trndata, trnseqlens) = dataset[:trn]
  (vlddata, vldseqlens) = dataset[:vld]
  (tstdata, tstseqlens) = dataset[:tst]
  cur_wkl, step, cur_lr = 0, 0, 0
  best_vld_cost = 100000
  for e = 1:o[:epochs]
    for i = 1:length(trndata)
      cur_lr = (LRP.lr - LRP.minlr)*(LRP.decayrate^step) + LRP.minlr
      x = perturb(trndata[i]; scalefactor=o[:scalefactor])
      grads = contextgrad(model, map(a->convert(atype, a), x), trnseqlens[i]; dprob=o[:dprob])
      updatelr!(opts, cur_lr)
      update!(model, grads, opts)
      step += 1
    end
    vld_rec_loss = evaluatemodel(model, vlddata, vldseqlens)
    #save the best model
    vld_cost = vld_rec_loss
    if vld_cost < best_vld_cost
      best_vld_cost = vld_cost
      arrmodel = convertmodel(model)
      println("Epoch: $(e) saving best model to $(pretrnp)$(o[:bestmodel])")
      save("$(pretrnp)$(o[:bestmodel])","model", arrmodel)
    end
    #report losses
    @printf("vld data - epoch: %d step %d rec loss: %g lr: %g \n", e, step, vld_rec_loss, cur_lr)
    #save every o[:save_every] epochs
    if e%o[:save_every] == 0
      arrmodel = convertmodel(model)
      save("$(pretrnp)m$(e)$(o[:tmpmodel])","model", arrmodel)
    end
    flush(STDOUT)
  end
end

function evaluatemodel(model, data, seqlens)
  rec_loss, kl_loss = 0, 0
  for i = 1:length(data)
    penstate_loss, offset_loss = bi_lstm_gen(model, map(a->convert(atype, a), data[i]), seqlens[i]; istraining = false)
    rec_loss += (penstate_loss + offset_loss)
  end
  return rec_loss/length(data)
end

function minibatch(sketchpoints3D, numbatches, params)
  info("Minibatching")
  data = []
  idm_data = []
  seqlens = []
  for i=0:(numbatches-1)
    x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, i, params)
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
