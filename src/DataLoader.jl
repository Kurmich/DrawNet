
include("Drawing.jl")
module DataLoader
using JSON
using Drawing

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
      push!(points, Int(stroke[1][i]))
      push!(points, Int(stroke[2][i]))
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

function main()
  dict2 = getsketches("full_simplified_airplane.ndjson")
  printcontents(dict2[25])
  initpointvocab(dict2)
end
main()
end
