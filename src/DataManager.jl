
function add_annotations!(dict_data, annotations)
  sketch = getsketch(dict_data)
  #add labels for each stroke
  for label in keys(annotations)
    if haskey(dict_data, label)
      points, end_indices = getstrokes(dict_data[label]) #get all strokes for label
      #add each stroke and its label to annotations
      for strokenum=1:(length(end_indices)-1)
        start_ind = end_indices[strokenum]+1
        end_ind = end_indices[strokenum+1]
        #get points of stroke
        ps = points[:, start_ind:end_ind]
        ends = [0, length(start_ind:end_ind)]
        push!(annotations[label], (ps, ends, sketch))
      end
    end
  end
end


function add_details!(dict_data, annotations)
  for label in keys(annotations)
    if haskey(dict_data, label)
      points, end_indices = getstrokes(dict_data[label])
      #label each sketch
      for strokenum=1:(length(end_indices)-1)
        start_ind = end_indices[strokenum]+1
        end_ind = end_indices[strokenum+1]
        #get points of stroke
        ps = points[:, start_ind:end_ind]
        ends = [0, length(start_ind:end_ind)]
        push!(annotations[label], (ps, ends))
      end
    #  push!(annotations[label], (points, end_indices))
    end
  end
end

function initannot(labels)
  annotations = Dict()
  for label in labels
    annotations[label] = []
  end
  return annotations
end

function getannotateddata(filename, labels)
  println("Airplane annotations")
  annot_dicts = [] #contains all of the data in dictionaty format
  annotations = initannot(labels) #annotations is the dictionary with label -> list of (ps(of stroke), ends(of stroke), sketch(this stroke belongs to))
  open(filename, "r") do f
     while !eof(f)
       text_data = readline(f)  # file information to string
       dict_data = JSON.parse(text_data)  # parse and transform data
       push!(annot_dicts, dict_data)
       add_annotations!(dict_data, annotations) ## CHANGE HERE
     end
  end
  return annotations, annot_dicts
end

function annot2pic(filename, labels)
  #gets annotations and saves them to images
  classnames = copy(labels)
  i = 1
  push!(classnames, "No Label")
  open(filename, "r") do f
     while !eof(f)
       text_data = readline(f)  # file information to string
       dict_data = JSON.parse(text_data)  # parse and transform data
       sketch = getsketch(dict_data)
       strokeclasses = zeros(1, (length(sketch.end_indices)-1) )
       for label in labels
         if haskey(dict_data, label)
           points, end_indices = getstrokes(dict_data[label])
           #label each sketch
           for strokenum=1:(length(end_indices)-1)
             start_ind = end_indices[strokenum]+1
             end_ind = end_indices[strokenum+1]
             #get points of stroke
             ps = points[:, start_ind:end_ind]
             for sk_strokenum=1:(length(sketch.end_indices)-1)
               sk_start_ind = sketch.end_indices[sk_strokenum]+1
               sk_end_ind = sketch.end_indices[sk_strokenum+1]
               #get points of stroke
               sk_ps = sketch.points[:, sk_start_ind:sk_end_ind]
               if length(sk_ps) == length(ps)
                 if sum(sk_ps .- ps) == 0
                   class = findfirst(labels, label)
                   strokeclasses[sk_strokenum] = class
                 end
               end
             end
           end
         end
       end
       strokeclasses[strokeclasses.==0] = length(classnames)
       saveslabeled(sketch, strokeclasses, classnames, "segmentedpics/huangairp$(i).png")
       i += 1
     end
  end
end


function getannotatedkeys(filename)
  println("Getting annotated data keys")
  akeys = Dict{String, Bool}()
  open(filename, "r") do f
     while !eof(f)
       text_data = readline(f)  # file information to string
       dict_data = JSON.parse(text_data)  # parse and transform data
       akeys[dict_data["key_id"]] = true
     end
  end
  return akeys
end

function annotated2sketch_obj(annotations)
  #=
  Returns
  sketch_objects(type: Dict) = label -> list of Sketch() objects for the labeled stroke
  full_sketch_objects(type: Dict) = label -> list of Sketch() objects for full sketch
  sketch_objects, full_sketch_objects correspond to each other in order
  =#
  sketch_objects = Dict()
  full_sketch_objects = Dict()
  for label in keys(annotations)
    if ! haskey(sketch_objects, label)
      sketch_objects[label] = []
      full_sketch_objects[label] = []
    end
    for (points, end_indices, sketch) in annotations[label]
      push!(sketch_objects[label], Sketch(label, true, sketch.key_id, points, end_indices))
      push!(full_sketch_objects[label], sketch)
    end
  end
  return sketch_objects, full_sketch_objects
end


function getrandannots(annot_dicts, labels, count)
  @assert(count <= length(annot_dicts), "Number of annotated data to use must be <= available data")
  trnannotations = initannot(labels)
  rp = randperm(length(annot_dicts))
  annot2use = annot_dicts[1:count]
  for dict_data in annot2use
    add_annotations!(dict_data, trnannotations)
  end
  annot2use = annot_dicts[count+1:end]
  tstannotations = initannot(labels)
  for dict_data in annot2use
    add_annotations!(dict_data, tstannotations)
  end
  return trnannotations
end

function dict2list(dictdata, labels)
  #=Puts all data in dictionary to list and returns=#
  listdata = []
  for label in labels
    append!(listdata, dictdata[label])
  end
  return listdata
end

function randindinces(annotations)
  indices = Dict()
  for label in keys(annotations)
    indices[label] = randperm(length(annotations[label]))
  end
  return indices
end


function shift_indices!(indices, vldsize)
  #=Circularly shifts indices by vldsize=#
  for label in keys(indices)
    idx = indices[label]
    up = Int(floor(vldsize*length(idx)))
    if up == 0
      continue
    end
    indices[label] = circshift(idx, up) #Circularly shift, i.e. rotate, the data in an array by "up" amount.
  end
end


function getshiftedindx(tt_indx, o)
  up = div(length(tt_indx), o[:cvfolds])
  println("shifting by $(up)")
  return circshift(tt_indx, up)
end

function fillannotations!(annotations, annot2use)
  for dict_data in annot2use
    add_annotations!(dict_data, annotations)
  end
end

function data_tt_split(annot_dicts, trncount::Int; rp = nothing)
  #=train test split based on sketches=#
  @assert(trncount > 0 && trncount < length(annot_dicts), "tstsize should be in range 0 < tstsize < $(length(annot_dicts))")
  rp = (rp == nothing) ? randperm(length(annot_dicts)) : rp
  trn_dicts = annot_dicts[rp[1:trncount]]
  tst_dicts = annot_dicts[rp[trncount+1:length(rp)]]
  return trn_dicts, tst_dicts
end

function getannotationdict(dict_data, labels)
  annotations = initannot(labels)
  fillannotations!(annotations, dict_data)
  return annotations
end

function train_test_split(data, tstsize; indices = nothing)
  #=Train test split based on components=#
  @assert(tstsize > 0 && tstsize < 1, "tstsize should be in range 0 < tstsize < 1")
  indices = (indices == nothing) ? randindinces(data) : indices
  trndata = Dict()
  tstdata = Dict()
  for label in keys(data)
    idx = indices[label]
    curdata = data[label]
    up = Int(floor(tstsize*length(idx))) # point of tst-trn split
    tstdata[label] = curdata[idx[1:up]] # 1:up indices are used for tesing
    trndata[label] = curdata[idx[up+1:length(idx)]] # up:end are used for training
  end
  return trndata, tstdata
end

function printdatastats(annotations)
  for label in keys(annotations)
    println("$(label) $(length(annotations[label]))")
  end
end

function getHuangLabels()
    category = Dict()
    category["airplane"]   = ["body", "wing", "horistab", "vertstab",  "engine", "propeller"]
    category["bicycle"]    = ["saddle", "frontframe", "wheel", "handle", "pedal", "chain", "fork", "backframe", "backcover" ]
    category["candelabra"] = ["base", "candle", "fire", "handle", "shaft", "arm"]
    category["chair"]      = ["back", "leg", "seat", "arm", "stile", "gas lift", "base", "foot", "stretcher", "spindle", "rail"]
    category["fourleg"]    = ["head", "body", "ear", "leg", "tail"]
    category["human"]      = ["head", "body", "arm", "leg", "hand", "foot"]
    category["lamp"]       = ["tube", "base", "shade"]
    category["rifle"]      = ["barrel", "body", "handgrip", "magazine", "trigger", "butt", "sight"]
    category["table"]      = ["top", "leg", "stretcher", "base", "topsupport", "legsupport", "midsupport"]
    category["vase"]       = ["lip", "handle", "body", "foot"]
    return category
end


function getGoogleLabels()
  category = Dict()
  category["airplane"]   = ["W", "B", "T", "WNDW", "FA"]
  category["cat"]        = [ "EAR", "H", "EYE", "N", "W", "M",  "B", "T", "L"]
  category["firetruck"]  = [ "LGT", "LDR", "B", "C", "WNDW", "WHS",  "WHL"]
  category["chair"]      = [ "B", "S", "L"]
  category["flower"]     = [ "P", "C" ,"S", "L"]
  return category
end
