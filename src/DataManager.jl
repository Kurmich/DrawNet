
function update_annotations!(dict_data, annotations)
  all_points, all_end_indices = getstrokes(dict_data["drawing"])
  all_strokes = []
  for strokenum=1:(length(all_end_indices)-1)
    start_ind = all_end_indices[strokenum]+1
    end_ind = all_end_indices[strokenum+1]
    #get points of stroke
    ps = all_points[:, start_ind:end_ind]
    push!(all_strokes, ps)
  end

  #add labels for each stroke
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
        #remove annotated stroke from pool of unannotated ones
        for ind= 1:length(all_strokes)
          if size(all_strokes[ind]) == size(ps) && isapprox(all_strokes[ind], ps)
          #  println("removing")
            deleteat!(all_strokes, ind)
            break
          end
        end
      end
    #  push!(annotations[label], (points, end_indices))
    end
  end
  label = "F"
  if haskey(dict_data, "FP")
    return
  end
  for ind = 1:length(all_strokes)
    if length(all_strokes[ind]) > 1
      push!(annotations[label], (all_strokes[ind], [0, size(all_strokes[ind], 2)]))
    else
      println("point")
    end
  end
end

function update_annotations_airplane!(dict_data, annotations)
  sketch = getsketch(dict_data)
  #add labels for each stroke
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
        push!(annotations[label], (ps, ends, sketch))
        #remove annotated stroke from pool of unannotated ones
      end
    #  push!(annotations[label], (points, end_indices))
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

function getannotateddata(filename, labels)
  println("Airplane annotations")
  annotations = Dict()
  for label in labels
    annotations[label] = []
  end
  open(filename, "r") do f
     while !eof(f)
       text_data = readline(f)  # file information to string
       dict_data = JSON.parse(text_data)  # parse and transform data
       update_annotations_airplane!(dict_data, annotations) ## CHANGE HERE
     end
  end
  return annotations
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
       saveslabeled(sketch, strokeclasses, classnames, "segmentedpics/orig$(i).png")
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

function dict2list(dictdata, labels)
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
    indices[label] = circshift(idx, up) #Circularly shift, i.e. rotate, the data in an array by "up" amount.
  end
end

function train_test_split(data, tstsize; indices = nothing)
  #=Train test split =#
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
