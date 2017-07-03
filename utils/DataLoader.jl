include("Drawing.jl")
module DataLoader
using JSON
using Drawing

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
Parameters(; batchsize=100, max_seq_length=45, min_seq_length=30, scalefactor=1.0, rand_scalefactor=0.0, augment_prob=0.0, limit=100, numbatches=1)=Parameters(batchsize, max_seq_length, min_seq_length, scalefactor, rand_scalefactor, augment_prob, limit,numbatches, nothing )

global const datapath = "../data/"
function initpointvocab(sketches)
  pointvocab = Dict{Tuple, Int}()
  count = 0
  for sketch in sketches
    for i=1:size(sketch.points, 2)
      x = sketch.points[1, i]
      y = sketch.points[2, i]
      if !haskey(pointvocab, (x,y))
        count += 1
        pointvocab[(x,y)] = count
      end
    end
  end
  info("Number of unique points $(count)")
  return pointvocab
end

function getstrokes(drawing)
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


function initsketch(sketch_dict::Dict)
  recognized = Bool(sketch_dict["recognized"])
  if !recognized
    return nothing #get only recognized sketches
  end
  drawing = sketch_dict["drawing"]
  label = String(sketch_dict["word"])
  key_id = String(sketch_dict["key_id"])
  points, end_indices = getstrokes(drawing)
  return Sketch(label, recognized, key_id, points, end_indices)
end

function getsketches(filename)
  sketches = []
  open(filename, "r") do f
     while !eof(f)
       sketch_dict = Dict()
       sketch_text = readline(f)  # file information to string
       sketch_dict=JSON.parse(sketch_text)  # parse and transform data
       sketch = initsketch(sketch_dict)
       if sketch != nothing
         push!(sketches, sketch)
       end
     end
  end
  return sketches
end

function getmaxlen(sketches)
  #=returns maximum length of points=#
  maxlen = 0
  for sketch in sketches
    maxlen = max(maxlen, size(sketch.points, 2))
  end
  return maxlen
end

function preprocess(sketches, params::Parameters)
  #=Remove sketches having > max_seq_length points or < min_seq_length=#
  rawpoints = []
  seqlen = Int[]
  sketchpoints3D = []
  countdata = 0
  for sketch in sketches
    points = points_to_3d(sketch)
    len = size(points, 2)
    if len <= params.max_seq_length && len > params.min_seq_length
      #points = to_big_points(points)
      countdata += 1
      #remove large gaps from data?
      points[1:2, :] /= params.scalefactor
      push!(rawpoints, points)
      push!(seqlen, len)
    end
  end
  #sorted order according to sequence lengths
  idx = sortperm(seqlen)
  for i=1:length(seqlen)
    push!(sketchpoints3D, rawpoints[idx[i]])
  end
  println("total images <= max_seq_length($(params.max_seq_length)) is $(countdata)")
  params.numbatches = div(countdata, params.batchsize)
  #=returns in stroke-3 format=#
  return sketchpoints3D, params.numbatches
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

function normalize!(sketchpoints3D, params::Parameters; scalefactor = nothing)
  #=Normalize entire dataset (delta_x, delta_y) by the scaling factor.=#
  scalefactor = (scalefactor == nothing)? get_scalefactor(sketchpoints3D) : scalefactor
  params.scalefactor = scalefactor
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

function constructsketch()
end

function padbatch(batch, params::Parameters)
  max_len = params.max_seq_length
  result = zeros(5, max_len + 1, params.batchsize)
  @assert(length(batch)==params.batchsize)
  for i=1:params.batchsize
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

function indices_to_batch(sketchpoints3D, indices, params::Parameters)
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
  x_batch_5D = padbatch(x_batch, params)
  return x_batch, x_batch_5D, seqlen
end

function getbatch(sketchpoints3D, idx, params::Parameters)
  @assert(idx >= 0, "index must be nonnegative")
  @assert(idx < params.numbatches, "index must be less number of batches")
  start_ind = idx * params.batchsize
  indices = (start_ind + 1):(start_ind + params.batchsize)
  return indices_to_batch(sketchpoints3D, indices, params)
end

function getsketchpoints3D(filename = "full_simplified_airplane.ndjson"; params::Parameters=Parameters())
  filepath = "$datapath$filename"
  info("Retrieving sketches from $(filepath) file")
  sketches = getsketches(filepath)
  info("Retrieving 3D points from sketches")
  sketchpoints3D, numbatches = preprocess(sketches, params)
  return sketchpoints3D, numbatches
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
  return sketchpoints3D[trnidx], sketchpoints3D[vldidx], sketchpoints3D[tstidx]
end

function test()
  params = Parameters()
  num = 5
  filename = "full_simplified_airplane.ndjson"
  filepath = "$datapath$filename"
  sketches = getsketches(filepath)
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
  #initpointvocab(sketches)
end
#test()
export getsketchpoints3D
export normalize!
export getbatch
export splitdata
export Parameters
end