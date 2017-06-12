
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
Parameters(; batchsize=100, max_seq_length=250, min_seq_length=10, scalefactor=1.0, rand_scalefactor=0.0, augment_prob=0.0, limit=100, numbatches=1)=Parameters(batchsize, max_seq_length, min_seq_length, scalefactor, rand_scalefactor, augment_prob, limit,numbatches, nothing )

global const datapath = "/mnt/kufs/scratch/kkaiyrbekov15/DrawNet/data/"
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
       push!(sketches, initsketch(sketch_dict))
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
  #=Remove sketches having > max_seq_length points=#
  rawpoints = []
  seqlen = Int[]
  sketchpoints = []
  countdata = 0
  for sketch in sketches
    points = points_to_3d(sketch)
    len = size(points, 2)
    if len <= params.max_seq_length && len > params.min_seq_length
      points = to_big_points(points)
      countdata += 1
      #remove large gaps from data?
      points[1:2, :] /= params.scalefactor
      push!(rawpoints, points)
      push!(seqlen, len)
    end
  end
  #sorted order
  idx = sortperm(seqlen)
  for i=1:length(seqlen)
    push!(sketchpoints, rawpoints[idx[i]])
  end
  println("total images <= max_seq_length is $(countdata)")
  params.numbatches = div(countdata, params.batchsize)
  return sketchpoints, params.numbatches
end

function get_scalefactor(sketchpoints; max_seq_length::Int=250)
  #=Calculate the normalizing scale factor.=#
  data = Float32[]
  for i=1:length(sketchpoints)
    points = sketchpoints[i]
    for j=1:size(points, 2)
      push!(data, points[1, j])
      push!(data, points[2, j])
    end
  end
  return std(data)
end

function normalize!(sketchpoints, params::Parameters; scalefactor = nothing)
  #=Normalize entire dataset (delta_x, delta_y) by the scaling factor.=#
  scalefactor = (scalefactor == nothing)? get_scalefactor(sketchpoints) : scalefactor
  params.scalefactor = scalefactor
  for points in sketchpoints
    points[1:2, :] /= scalefactor
  end
end

function getbatch(idx, params::Parameters)
  @assert(idx >=1, "index must be positive")
  @assert(idx <= params.numbatches, "index must be less than batchsize")
end

function test()
  params = Parameters()
  num = 5
  filename = "full_simplified_airplane.ndjson"
  filepath = "$datapath$filename"
  sketches = getsketches(filepath)
  println("max_len=$(getmaxlen(sketches))")
  sketchpoints, numbatches = preprocess(sketches, params)
  @assert(params.numbatches != nothing && numbatches==params.numbatches)
  println(get_scalefactor(sketchpoints))
  println("normalizing")
  copy_sketchpoints = deepcopy(sketchpoints)
  println(copy_sketchpoints[1] == sketchpoints[1])
  normalize!(sketchpoints, params)
  println(copy_sketchpoints[1] == sketchpoints[1])
  printcontents(sketches[num])
  savesketch(sketches[num])
  initpointvocab(sketches)
  getbatch(1, params)
end
test()
end
