module Drawing
using PyPlot
#sketch type
type Sketch
  label::String
  recognized::Bool
  key_id::String
  points::Array
  end_indices::Array
end


function stroke_points_to_3d(sketch::Sketch)
  #= Convert polyline format to 3-stroke format.=#
  pen_state = zeros(1, size(sketch.points, 2))
  #set end points states to 1 discard initial 0
  pen_state[sketch.end_indices[2:end]] = 1
  points = hcat([0; 0; 0], vcat(sketch.points, pen_state))
  result = nothing
  #compute (deta_x, delta_y)'s
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    points[1:2, start_ind+1:end_ind] -= points[1:2, start_ind:end_ind-1]
    if result == nothing
      result = points[:, start_ind+1:end_ind]
    else
      result = hcat(result, points[:, start_ind+1:end_ind])
    end
  end
  #discard first entry
  #printpoints(points[:, 2:len])
  return result
end


function points_to_3d(sketch::Sketch)
  #= Convert polyline format to 3-stroke format.=#
  pen_state = zeros(1, size(sketch.points, 2))
  #set end points states to 1 discard initial 0
  pen_state[sketch.end_indices[2:end]] = 1
  points = hcat([0; 0; 0], vcat(sketch.points, pen_state))
  #compute (deta_x, delta_y)'s
  len = size(points, 2)
  points[1:2, 2:len] -= points[1:2, 1:len-1]
  #discard first entry
  #printpoints(points[:, 2:len])
  return points[:, 2:len]
end

function to_big_points(points; max_len=250)
  #=Make this sketchpoints the special bigger format as described in sketch-rnn paper.=#
  len = size(points, 2)
  rows = size(points, 1)
  @assert(rows == 3 && len <= max_len)
  result = zeros(5, max_len)
  #copy delta_x and delta_y
  result[1:2, 1:len] = points[1:2, :]
  #set state p2
  result[4, 1:len] = points[3, :]
  #set state p1
  result[3, 1:len] = 1 - result[4, 1:len]
  #set state p3
  result[5, len+1:max_len] = 1
  #printpoints(result)
  return result
end


function isendstroke(p, i)
  return Int(floor(p[1, i])) == 0 && Int(floor(p[2, i])) == 0 && Int(p[3, i]) == 0 && Int(p[4, i]) == 1 && Int(p[5, i]) == 0
end

function stroke_clean_points(points; factor::Int=100)
  #=Cut irrelevant end points, scale to pixel space and store as integer.=#
  len = size(points, 2)
  rows = size(points, 1)
  @assert(rows == 5)
  copy_points = nothing
  added_final = false
  printpoints(points)
  #iterate through all points
  for i=1:len
    finish_flag = Int(points[5, i])
    if finish_flag == 0
      x = Int(floor(points[1, i]*factor))
      y = Int(floor(points[2, i]*factor))
      p1 = Int(points[3, i])
      p2 = Int(points[4, i])
      if copy_points == nothing
        copy_points = [x; y; p1; p2; finish_flag]
      else
        copy_points = hcat(copy_points, [x; y; p1; p2; finish_flag])
      end
    else
      copy_points = hcat(copy_points, [0; 0; 0; 0; 1])
      added_final = true
      break
    end
  end
  if !added_final
    copy_points = hcat(copy_points, [0; 0; 0; 0; 1])
  end
  printpoints(copy_points)
  return copy_points
end

function clean_points(points; factor::Int=100)
  #=Cut irrelevant end points, scale to pixel space and store as integer.=#
  len = size(points, 2)
  rows = size(points, 1)
  @assert(rows == 5)
  copy_points = nothing
  added_final = false
  #iterate through all points
  for i=1:len
    finish_flag = Int(points[5, i])
    if finish_flag == 0
      x = Int(floor(points[1, i]*factor))
      y = Int(floor(points[2, i]*factor))
      p1 = Int(points[3, i])
      p2 = Int(points[4, i])
      if copy_points == nothing
        copy_points = [x; y; p1; p2; finish_flag]
      else
        copy_points = hcat(copy_points, [x; y; p1; p2; finish_flag])
      end
    else
      copy_points = hcat(copy_points, [0; 0; 0; 0; 1])
      added_final = true
      break
    end
  end
  if !added_final
    copy_points = hcat(copy_points, [0; 0; 0; 0; 1])
  end
  printpoints(copy_points)
  return copy_points
end

function printpoints(points)
  #=prints points in 3d=#
  info("Printing points")
  for i=1:size(points, 2)
    println(points[:, i])
  end
end

function addstroke!(sketch::Sketch, stroke::Array)
  #add strokes to sketch and add ending index of the stroke
  sketch.points = hcat(sketch.points, stroke)
  #if this is first stroke of sketch
  if length(sketch.end_indices) == 0
    push!(sketch.end_indices, size(stroke, 2))
  else
    push!(sketch.end_indices, sketch.end_indices[end] + size(stroke, 2))
  end
end

function printcontents(sketch::Sketch)
  #Prints contents of current sketch
  println("Contents of sketch with ID = $(sketch.key_id) are:")
  #iterate through strokes
  for strokenum= 1:(length(sketch.end_indices)-1)
    println("Points of stroke $strokenum")
    #print points of stroke
    start_ind = sketch.end_indices[strokenum] + 1
    end_ind = sketch.end_indices[strokenum+1]
    for i = start_ind:end_ind
      println("x = $(sketch.points[1, i]) y = $(sketch.points[2, i])")
    end
  end
end

function plotsketch(sketch::Sketch)
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    x = sketch.points[1, start_ind:end_ind]
    y = sketch.points[2, start_ind:end_ind]
    plot(x, y, linewidth = 1)
  end
end

function savesketch(sketch::Sketch, filename::String="sketch.png"; completness=1, mydpi=100, imsize=256, scaled = false)
  #Prints contents of current sketch
  fig = figure(figsize=(imsize/mydpi, imsize/mydpi), dpi=mydpi)
  strokelimit = Integer(ceil(completness*length(sketch.end_indices)))
  #iterate through strokes
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    #get points of stroke
    x = sketch.points[1, start_ind:end_ind]
    y = sketch.points[2, start_ind:end_ind]
    if strokenum <= strokelimit
      plot(x, -y, linewidth = 1)
    else
      plot(x, -y, linewidth = 1, color = "white")
    end
    if scaled
      subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)
    end
    axis("off")
  end
  savefig(filename, dpi=mydpi)
close()
end
export Sketch
export RNNSketch
export printcontents
export addstroke!
export plotsketch
export savesketch
export points_to_3d
export to_big_points
export clean_points, stroke_clean_points
end
