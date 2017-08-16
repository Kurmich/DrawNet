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

function trainsvm(annotations)
  CS = [1.0 2.0 4.0 8.0 16.0 1e2 5e2 1e3]
  GAMMAS = [0.5 0.25 0.125]
  tstacc = 0
  for C in CS
    for gamma in GAMMAS
      svmmodel = SVR.train(trnlabels, trnfeats; svm_type=SVR.C_SVC, kernel_type=SVR.RBF, C=C, gamma=gamma)
      acc = getaccuracy(svmmodel, tstfeats, tstlabels)
      tstacc = max(acc, tstacc)
      println("C=$(C), tst acc=$(tstacc)")
      SVR.freemodel(svmmodel)
    end
  end
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="SVM TRAINER. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
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
  getfeats(annotations)
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "SVMTrainer.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end
