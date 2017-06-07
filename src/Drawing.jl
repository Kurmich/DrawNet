module Drawing

#sketch type
type Sketch
  label::String
  recognized::Bool
  key_id::String
  points::Array
  end_indices::Array
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
export Sketch
export printcontents
export addstroke!
end
