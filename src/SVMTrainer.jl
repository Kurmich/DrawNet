include("../utils/DataLoader.jl")
include("../idm/IDM.jl")

module SVMTrainer
using Drawing, DataLoader, IDM
using JSON, SVR, ArgParse

function update_annotations!(dict_data, annotations)
  for label in keys(annotations)
    if haskey(dict_data, label)
      points, end_indices = getstrokes(dict_data[label])
      push!(annotations[label], (points, end_indices))
    end
  end
end

function getdata(filename, labels)
  annotations = Dict()
  for label in labels
    annotations[label] = []
  end
  open(filename, "r") do f
     while !eof(f)
       text_data = readline(f)  # file information to string
       dict_data = JSON.parse(text_data)  # parse and transform data
       update_annotations!(dict_data, annotations)
     end
  end
  return annotations
end

function printdatastats(annotations)
  for label in keys(annotations)
    println("$(label) $(length(annotations[label]))")
  end
end

function getfeats(annotations)
  info("Extracting idm features")
  features = Dict()
  for label in keys(annotations)
  #  println("$(label) $(length(annotations[label]))")
    features[label] = []
    for (points, end_indices) in annotations[label]
      push!(features[label], extractidm(points, end_indices))
    end
  end
  return features
end

function getaccuracy(svmmodel, features, ygold)
  ypred = SVR.predict(svmmodel, features)
  count = 0.0
  for i=1:length(ypred)
    if ypred[i] == ygold[i]
      count += 1
    end
  end
  return count/length(ypred)
end

function randindinces(annotations)
  indices = Dict()
  for label in keys(annotations)
    indices[label] = randperm(length(annotations[label]))
  end
  return indices
end

function train_test_split(data, tstsize; indices = nothing)
  @assert(tstsize > 0 && tstsize < 1, "tstsize should be in range 0 < tstsize < 1")
  indices = (indices == nothing) ? randindinces(data) : indices
  trndata = Dict()
  tstdata = Dict()
  for label in keys(data)
    idx = indices[label]
    curdata = data[label]
    up = Int(floor(tstsize*length(idx)))
    tstdata[label] = curdata[idx[1:up]]
    trndata[label] = curdata[idx[up+1:length(idx)]]
  end
  return trndata, tstdata
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
    acc = getaccuracy(svmmodel, vldfeats, vldlabels)
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
    end
  end
  @printf("best C: %g best gamma: %g best avg score: %g \n", bestC, bestGamma,  bestscore)
  return bestC, bestGamma
end

function trainsvm(trndata, tstdata, C, gamma, labels)
  trnfeats, trnlabels = get_feats_and_classes(trndata, labels)
  tstfeats, tstlabels = get_feats_and_classes(tstdata, labels)
  svmmodel = SVR.train(trnlabels, trnfeats; svm_type=SVR.C_SVC, kernel_type=SVR.RBF, C=C, gamma=gamma)
  acc = getaccuracy(svmmodel, tstfeats, tstlabels)
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
    ("--datapath"; arg_type=String; default="../annotateddata/"; help="Number of epochs per checkpoint creation.")
    ("--filename"; arg_type=String; default="r_full_simplified_airplane.ndjson"; help="Decoder: lstm, or ....")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  filename = string(o[:datapath], o[:filename])
  labels = [ "UpW", "LoW", "F", "FWSR", "FWSL", "LS", "RS","LW", "RW", "O"]
  annotations = getdata(filename, labels)
  printdatastats(annotations)
  idms = getfeats(annotations)
  trnidms, tstidms = train_test_split(idms, o[:tstsize])
  C, gamma = get_cvd_params(trnidms, o[:cv], labels)
  trainsvm(trnidms, tstidms, C, gamma, labels)
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "SVMTrainer.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end
