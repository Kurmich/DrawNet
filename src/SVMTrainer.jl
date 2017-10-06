include("../utils/DataLoader.jl")
include("../idm/IDM.jl")


module SVMTrainer
include("DataManager.jl")
using Drawing, DataLoader, IDM
using JSON, SVR, ArgParse, JLD

function getfeats(annotations)
  info("Extracting idm features")
  features = Dict()
  for label in keys(annotations)
  #  println("$(label) $(length(annotations[label]))")
    features[label] = []
    for (points, end_indices, sketch) in annotations[label]
      mid = sum(points, 2)/(size(points, 2)*256) ##ADDING SPATIAL INFO
      idm = extractidm(points, end_indices)
      #println(size(idm), size(mid))
      idm = hcat(idm, mid')
      idm = hcat(idm, points[:, 1]'/256)
      idm = hcat(idm, points[:, size(points,2)]'/256)
      push!(features[label], idm)
    end
  end
  return features
end


function getaccuracy(svmmodel, features, ygold)
  ypred = SVR.predict(svmmodel, features)
  @assert(length(ypred) == length(ygold))
  correct_count = zeros(1, length(unique(ygold)))
  instance_count = zeros(1, length(unique(ygold)))
  count = 0.0
  for i=1:length(ypred)
    #println(ypred[i])
    #println(ygold[i])
    class = Int(ygold[i])
    instance_count[class] += 1
    if ypred[i] == ygold[i]
      correct_count[class] += 1
      count += 1
    end
  end
  return count/length(ypred), correct_count, instance_count
end


function shift_indices!(indices, vldsize)
  for label in keys(indices)
    idx = indices[label]
    up = Int(floor(vldsize*length(idx)))
    indices[label] = circshift(idx, up)
  end
end


function get_feats_and_classes(data, labels)
  feats = nothing
  classes = []
  for i = 1:length(labels)
    key = labels[i]
    for feat in data[key]
      if feats == nothing
        feats = feat
      else
        feats = vcat(feats, feat)
      end
      push!(classes, i)
    end
  end
  #info("Size of feats = $(size(feats)) classes = $(size(classes))")
  classes = convert(Array{Float64}, classes)
  return feats', classes
end

function crossvalidate(data, cv, labels, C, gamma)
  @assert(cv > 1, "Cross validation parameter should be greater than 1")
  #info("Cross validating for C = $(C) and gamma = $(gamma)")
  indices = randindinces(data)
  vldsize = 1/cv
  scores = []
  for i = 1:cv
    shift_indices!(indices, vldsize)
    trndata, vlddata = train_test_split(data, vldsize; indices = indices)
    trnfeats, trnlabels = get_feats_and_classes(trndata, labels)
    vldfeats, vldlabels = get_feats_and_classes(vlddata, labels)
    svmmodel = SVR.train(trnlabels, trnfeats; svm_type=SVR.C_SVC, kernel_type=SVR.RBF, C=C, gamma=gamma)
    acc, correct_count, instance_count = getaccuracy(svmmodel, vldfeats, vldlabels)
    println(correct_count ./ instance_count)
    push!(scores, acc)
    SVR.freemodel(svmmodel)
  end
  return scores
end

function get_cvd_params(data, cv, labels)
  #search for best cross validation parameters
  CS = []
  GAMMAS = []
  for i=1:11
    push!(CS, 2.0^i)
    push!(GAMMAS, 2.0^(-i))
  end
  bestC, bestGamma, bestscore = 0, 0, 0
  for C in CS
    for gamma in GAMMAS
      scores = crossvalidate(data, cv, labels, C, gamma)
      avgscore = sum(scores) / length(scores)
      if avgscore > bestscore
        bestC = C
        bestGamma = gamma
        bestscore = avgscore
      end
      println(scores)
      @printf("C: %g gamma: %g avgscore: %g \n \n", C, gamma,  avgscore)
      flush(STDOUT)
    end
  end
  @printf("best C: %g best gamma: %g best avg score: %g \n", bestC, bestGamma,  bestscore)
  return bestC, bestGamma
end

function trainsvm(trndata, tstdata, C, gamma, labels)
  trnfeats, trnlabels = get_feats_and_classes(trndata, labels)
  tstfeats, tstlabels = get_feats_and_classes(tstdata, labels)
  svmmodel = SVR.train(trnlabels, trnfeats; svm_type=SVR.C_SVC, kernel_type=SVR.RBF, C=C, gamma=gamma)
  acc, correct_count, instance_count = getaccuracy(svmmodel, tstfeats, tstlabels)
  @printf("best C: %g best gamma: %g tst accuracy: %g \n", C, gamma, acc)
  SVR.savemodel(svmmodel, "airplane.model")
  SVR.freemodel(svmmodel)
end


function main(args=ARGS)
  s = ArgParseSettings()
  s.description="SVM model trainer. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--tstsize"; arg_type=Float64; default=0.2; help="Test set proportion.")
    ("--cv"; arg_type=Int; default=5; help="Cross validation fold parameter.")
    ("--datapath"; arg_type=String; default="../annotateddata/"; help="Path to annotated data.")
    ("--filename"; arg_type=String; default="r_full_simplified_airplane.ndjson"; help="Filename of annotated data.")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  filename = string(o[:datapath], o[:filename])
  labels = [  "W", "B", "T" ,"WNDW", "FA"]
  #labels = [ "L", "F", "FP"]
  vldsize = 1/5
  annotations = getannotateddata(filename, labels)
  #=sketches = annotated2sketch_obj(annotations)
  params = Parameters()
  indices = randindinces(sketches)
  trndict, tstdict = train_test_split(sketches, vldsize; indices = indices) #get even split as dictionary
  indices = randindinces(trndict)
  trndict, vlddict = train_test_split(trndict, vldsize; indices = indices)
  trndata = dict2list(trndict)  #as list ,> this is list of lists we need just list of sketches
  vlddata = dict2list(vlddict)
  tstdata = dict2list(tstdict) #as list
  sketchpoints3D, numbatches, sketches = preprocess(trndata, params) =#

  #println(numbatches)
  printdatastats(annotations)
  idms = getfeats(annotations)
  if o[:readydata]
    trn_tst_indices = load("trn_tst_indices.jld")["indices"]
  else
    trn_tst_indices = nothing
  end
  trnidms, tstidms = train_test_split(idms, o[:tstsize]; indices = trn_tst_indices)
  C, gamma = get_cvd_params(trnidms, o[:cv], labels)
  trainsvm(trnidms, tstidms, C, gamma, labels)
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "SVMTrainer.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end
