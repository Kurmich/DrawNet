include("Drawing.jl")
module DataLoader
using JSON
using Drawing
include("../src/DataManager.jl")
type Parameters
  batchsize::Int
  max_seq_length::Int
  min_seq_length::Int
  scalefactor::AbstractFloat
  rand_scalefactor::AbstractFloat
  augment_prob::AbstractFloat
  limit::Int
  numbatches::Int
  sketchpoints
end
Parameters(; batchsize=100, max_seq_length=90, min_seq_length=5, scalefactor=1.0, rand_scalefactor=0.10, augment_prob=0.0, limit=100, numbatches=1) = Parameters(batchsize, max_seq_length, min_seq_length, scalefactor, rand_scalefactor, augment_prob, limit,numbatches, nothing )

global const datapath = "../data/"
global const annotp = "../annotateddata/"

function getstrokes(drawing)
  #=returns all points in a drwing with end_indices as (points, end_indices) tuple=#
  points = Int[]
  end_indices = Int[]
  #starting index of first stroke is 0+1=1
  push!(end_indices, 0)
  for stroke in drawing
    #num of points in current stroke
    numpoints = length(stroke[1])
    for i = 1:numpoints
      push!(points, Float32(stroke[1][i]))
      push!(points, Float32(stroke[2][i]))
    end
    #update end index of current stroke
    push!(end_indices, end_indices[end] + numpoints)
  end
  points = reshape(points, (2, convert(Int64, length(points)/2)))
  #println("$(size(points)) $(size(end_indices))")
  return points, end_indices
end


function getsketch(sketch_as_dict::Dict)
  recognized = Bool(sketch_as_dict["recognized"])
  if !recognized
    return nothing #get only recognized sketches
  end
  drawing = sketch_as_dict["drawing"]
  label = String(sketch_as_dict["word"])
  key_id = String(sketch_as_dict["key_id"])
  points, end_indices = getstrokes(drawing)
  return Sketch(label, recognized, key_id, points, end_indices)
end

function get_sketch_objects(filename; a_filename = "nothing")
  println("Annotated file $(a_filename)")
  #akeys = getannotatedkeys(string(annotp, a_filename)) #SKIP ANNOTATED ONES
  akeys = Dict()

  sketch_objects = Sketch[]
  open(filename, "r") do f
     while !eof(f)
       sketch_as_dict = Dict()
       sketch_as_text = readline(f)  # file information to string
       sketch_as_dict = JSON.parse(sketch_as_text)  # parse and transform data
       if haskey(akeys, sketch_as_dict["key_id"])
         println("skipping $(sketch_as_dict["key_id"])")
         continue #SKIP ANNOTATED ONES
       end
       sketch = getsketch(sketch_as_dict)
       if sketch != nothing
         push!(sketch_objects, sketch)
       end
     end
  end
  return sketch_objects
end

function getmaxlen(sketch_objects)
  #=returns maximum length of points=#
  maxlen = 0
  for sketch in sketch_objects
    maxlen = max(maxlen, size(sketch.points, 2))
  end
  return maxlen
end

function preprocess_strokes(sketch_objects, params::Parameters)
  #=Remove sketches having > max_seq_length points or < min_seq_length=#
  rawstrokes = []
  seqlen = Int[]
  strokepoints3D = []
  filtered_sketches = []
  countdata = 0
  for sketch in sketch_objects
    strokes = get_3d_strokes(sketch) #HERE STROKE POINTS
    #prep each stroke of the sketch
    for stroke in strokes
      countdata += 1
      len = size(stroke, 2)
      if len > 50
        continue
      end
      #remove large gaps from data?
      #stroke[1:2, :] /= params.scalefactor
      push!(rawstrokes, stroke)
      #add stroke for corresponding sketch
      push!(filtered_sketches, sketch)
      push!(seqlen, len)
    end
  end
  #sorted order according to sequence lengths
  idx = sortperm(seqlen)
  sketches = []
  for i=1:length(seqlen)
  #  push!(strokepoints3D, rawstrokes[idx[i]])
  #  push!(sketches, filtered_sketches[idx[i]])
  end
  println("Total images <= max_seq_length($(params.max_seq_length)) is $(countdata)")
  params.numbatches = div(countdata, params.batchsize)
  #=returns in stroke-3 format=#
  return rawstrokes, params.numbatches, filtered_sketches
end

function getstrokedata(filename = "full_simplified_airplane.ndjson"; params::Parameters=Parameters())
  strokedata = Dict()
  filepath = "$datapath$filename"
  println("Retrieving sketches from $(filepath) file")
  sketches = get_sketch_objects(filepath; a_filename = a_filename)
  trnidx, vldidx, tstidx = splitdata(sketches)
  println("Retrieving 3D strokes from sketches")
  strokedata[:trn] = preprocess_strokes(sketches[trnidx], params) #returns trn_strokes3D, trn_numbatches, trn_sketches
  strokedata[:vld] = preprocess_strokes(sketches[vldidx], params) #returns vld_strokes3D, vld_numbatches, vld_sketches
  strokedata[:tst] = preprocess_strokes(sketches[tstidx], params) #return tst_strokes3D, tst_numbatches, tst_sketches
  strokedata[:idx] = (trnidx, vldidx, tstidx)
  strokedata[:sketches] = sketches
  return strokedata
end

function preprocess(sketch_objects, params::Parameters; maxdatacount = 0)
  #=Remove sketches having > max_seq_length points or < min_seq_length=#
  rawpoints = []
  seqlen = Int[]
  sketchpoints3D = []
  filtered_sketches = []
  countdata = 0
  for sketch in sketch_objects
    points = stroke_points_to_3d(sketch) #HERE STROKE POINTS
    len = size(points, 2)
    if len <= params.max_seq_length && len > params.min_seq_length
      #points = to_big_points(points)
      countdata += 1
      #remove large gaps from data?
      #points[1:2, :] /= params.scalefactor
      push!(rawpoints, points)
      push!(filtered_sketches, sketch)
      push!(seqlen, len)
    else
      println("skipping")
    end
  end
  #sorted order according to sequence lengths
  if maxdatacount != 0
    countdata = maxdatacount
    rawpoints = rawpoints[1:maxdatacount]
    filtered_sketches = filtered_sketches[1:maxdatacount]
  end
  println("total images <= max_seq_length($(params.max_seq_length)) is $(countdata)")
  params.numbatches = div(countdata, params.batchsize)
  #=returns in stroke-3 format=#
  return rawpoints, params.numbatches, filtered_sketches
end

function get_scalefactor(sketchpoints3D; max_seq_length::Int=250)
  #=Calculate the normalizing scale factor.=#
  data = Float32[]
  for i=1:length(sketchpoints3D)
    points = sketchpoints3D[i]
    for j=1:size(points, 2)
      push!(data, points[1, j])
      push!(data, points[2, j])
    end
  end
  return std(data)
end


function isnormalized(sketchpoints3D)
  stdev = get_scalefactor(sketchpoints3D)
  return 0.97 <= stdev && stdev <= 1.03
end

function normalize!(sketchpoints3D, params::Parameters; scalefactor = nothing)
  #=Normalize entire dataset (delta_x, delta_y) by the scaling factor.=#
  scalefactor = (scalefactor == nothing) ? get_scalefactor(sketchpoints3D) : scalefactor
  params.scalefactor = scalefactor
  println("Normalizing by $(scalefactor)")
  for points in sketchpoints3D
    points[1:2, :] /= scalefactor
  end
  return sketchpoints3D
end

function restore(sketchpoints3D, scalefactor = nothing)
  @assert(scalefactor != nothing, "Scale factor can't be nothing")
  for points in sketchpoints3D
    points[1:2, :] *= scalefactor
  end
  return sketchpoints3D
end


function padbatch_4d(batch, params::Parameters)
  max_len = params.max_seq_length
  result = zeros(4, max_len + 1, length(batch))
  #@assert(length(batch)==params.batchsize)
  for i=1:length(batch)
    len = size(batch[i], 2)
    @assert(len <= max_len)
    result[:, 2:max_len+1, i] = to_4d_points(batch[i]; max_len = max_len)
    #put in the first token, as described in sketch-rnn methodology
    result[1, 1, i] = 0
    result[2, 1, i] = 0
    result[3, 1, i] = 1
    result[4, 1, i] = 0
  #  printpoints(result[:, :, i])
  end
  return result
end

function padbatch(batch, params::Parameters)
  max_len = params.max_seq_length
  result = zeros(5, max_len + 1, length(batch))
  #@assert(length(batch)==params.batchsize)
  for i=1:length(batch)
    len = size(batch[i], 2)
    @assert(len <= max_len)
    result[:, 2:max_len+1, i] = to_big_points(batch[i]; max_len = max_len)
    #put in the first token, as described in sketch-rnn methodology
    result[1, 1, i] = 0
    result[2, 1, i] = 0
    result[3, 1, i] = 1
    result[4, 1, i] = 0
    result[5, 1, i] = 0
  end
  return result
end

function indices_to_batch(sketchpoints3D, indices, V, params::Parameters)
  x_batch = []
  seqlen = Int[]
  for idx=indices
    data  = sketchpoints3D[idx] #randomscale(sketchpoints[idx])
    data_copy = copy(data)
    if params.augment_prob > 0
      #perform augmentation
    end
    len = size(data_copy, 2)
    push!(x_batch, data_copy)
    push!(seqlen, len)
  end
  max_len = maximum(seqlen)
  old_max = params.max_seq_length
  params.max_seq_length = max_len + 1 #not to overpad
  if V == 4
    x_batch_5D = padbatch(x_batch, params)# padbatch_4d(x_batch, params)
  else
    x_batch_5D = padbatch(x_batch, params)
  end
  params.max_seq_length = old_max
  return x_batch, x_batch_5D, seqlen
end

function getbatch(sketchpoints3D, idx, V, params::Parameters)
  @assert(idx >= 0, "index must be nonnegative")
  @assert(idx < params.numbatches, "index must be less number of batches")
  start_ind = idx * params.batchsize
  end_ind = min((start_ind + params.batchsize), length(sketchpoints3D))
  indices = (start_ind + 1) : end_ind
  return indices_to_batch(sketchpoints3D, indices, V, params)
end

function getsketchpoints3D(filename = "full_simplified_airplane.ndjson"; a_filename = "nothing", params::Parameters=Parameters())
  data = Dict()
  filepath = "$datapath$filename"
  info("Retrieving sketches from $(filepath) file")
  sketches = get_sketch_objects(filepath; a_filename = a_filename)
  trnidx, vldidx, tstidx = splitdata(sketches)
  info("Retrieving 3D points from sketches")
  data[:trn] = preprocess(sketches[trnidx], params; maxdatacount = 70000) #trn_sketchpoints3D, trn_numbatches, trn_sketches
  data[:vld] = preprocess(sketches[vldidx], params; maxdatacount = 2500) #vld_sketchpoints3D, vld_numbatches, vld_sketches
  data[:tst] = preprocess(sketches[tstidx], params; maxdatacount = 2500) #tst_sketchpoints3D, tst_numbatches, tst_sketches
  data[:idx] = (trnidx, vldidx, tstidx)
  return data
end

#splits data to train, validation and test sets
function splitdata(sketchpoints3D; trn = 0.9, vld = 0.05, tst=0.05)
  perm = randperm(length(sketchpoints3D)) #random permutation
  #number of sketches in each slit
  trncount = Int(ceil(trn * length(sketchpoints3D)))
  vldcount = Int(ceil(vld * length(sketchpoints3D)))
  tstcount = Int(ceil(tst * length(sketchpoints3D)))
  #indices of sketches
  start = 1
  trnidx = perm[1:trncount]
  start += trncount
  vldidx = perm[start:start+vldcount]
  start += vldcount
  tstidx = perm[start:min(start + tstcount, length(sketchpoints3D))]
  #return trn, vld, tst sketch datasets
  return trnidx, vldidx, tstidx
end

function test()
  params = Parameters()
  num = 5
  filename = "full_simplified_airplane.ndjson"
  filepath = "$datapath$filename"
  sketches = get_sketch_objects(filepath)
  println("max_len=$(getmaxlen(sketches))")
  sketchpoints3D, numbatches = preprocess(sketches, params)
  trndata, vlddata, tstdata = splitdata(sketchpoints3D)
  @printf("training set: %g validation set: %g test set: %g sizes", length(trndata), length(vlddata), length(tstdata))
  x_batch, x_batch_5D, seqlen = getbatch(sketchpoints3D, 1, params)
  for len in seqlen
    println(len)
  end
  println("minibatch size = $(size(x_batch_5D[:, 2, :]))")
  @assert(params.numbatches != nothing && numbatches==params.numbatches)
  println(get_scalefactor(sketchpoints3D))
  println("normalizing")
  copy_sketchpoints= deepcopy(sketchpoints3D)
  println(copy_sketchpoints[1] == sketchpoints3D[1])
  normalize!(sketchpoints3D, params)
  println(copy_sketchpoints[1] == sketchpoints3D[1])
  #printcontents(sketches[num])
  #savesketch(sketches[num])
end

function main(args=ARGS)
  test()
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "DataLoader.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
export getsketchpoints3D, get_sketch_objects
export normalize!, isnormalized
export getbatch, getstrokes, getsketch, getstrokedata
export splitdata, preprocess
export Parameters
end
