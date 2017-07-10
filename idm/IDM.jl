include("../utils/DataLoader.jl")
#include("../utils/Drawing.jl")
module IDM
using DataLoader, Drawing
using ArgParse, PyPlot



function normalize(points; d = 2)
  center = mean(points, d)
  rstd = 1 ./ std(points, d)
  return (points .- center) .* rstd
end

function maxdist(points)
  #=find distance of furthest point from center of sketch=#
  center = mean(points, 2)
  diff = points .- center
  return maximum(sqrt(sumabs2(diff, 1)))
end

function resample_sketch(sketch::Sketch; coeff = 1.01, ratio = 50)
  interval = ( coeff * maxdist(sketch.points) ) / ratio #spatial samling interval
  end_indices = Int[]
  points = nothing
  #starting index of first stroke is 0+1=1
  push!(end_indices, 0)
  for strokenum=1:(length(sketch.end_indices)-1)
    start_ind = sketch.end_indices[strokenum]+1
    end_ind = sketch.end_indices[strokenum+1]
    newstroke, numpoints = resample_points(sketch.points; coeff = coeff, ratio = ratio, interval=interval, indices = start_ind:end_ind)
    if points == nothing
      points = newstroke
    else
      points = hcat(points, newstroke)
    end
    push!(end_indices, end_indices[end] + numpoints)
  end
  return points, end_indices
end

function resample_points(points::Array; coeff = 1.01, ratio = 50, indices = 1:size(points,2), interval = nothing)
  #=points[indices] -> all points of a single stroke=#
  if interval == nothing
    interval = ( coeff * maxdist(points) ) / ratio #spatial samling interval
  end
  numpoints = 0
  newpoints = nothing
  for i = indices[2] : indices[end]
    prev_point = points[1:2, i-1]
    cur_point = points[1:2, i]
    sampdistance = sqrt(sumabs2(cur_point .- prev_point))
    if newpoints == nothing
      newpoints = [prev_point[1]; prev_point[2]]
    else
      newpoints = hcat(newpoints, [prev_point[1]; prev_point[2]])
    end
    numpoints += 1
    while sampdistance > interval
      angle = atan2(cur_point[2] - prev_point[2], cur_point[1] - prev_point[1]) #LOOK FOR SIGNS HERE: mAYBE -Y ?
      x = prev_point[1] + interval * cos(angle)
      y = prev_point[2] + interval * sin(angle)
      prev_point = [x; y]
      newpoints = hcat(newpoints, prev_point)
      sampdistance = sqrt(sumabs2(cur_point .- prev_point))
      numpoints += 1
    end
  end
  println("numpoints=$(numpoints) size newpoints = $(size(newpoints))")
  return newpoints, numpoints
end


function coords2angles(points, end_indices)
  thetas = nothing
  thetaidx = [0]
  for strokenum=1:(length(end_indices)-1)
    start_ind = end_indices[strokenum]+1
    end_ind = end_indices[strokenum+1]
    diff = points[1:2, (start_ind+1): end_ind] - points[1:2, start_ind : (end_ind-1)]
    theta = (atan2(diff[1, :], diff[2, :]) % 2pi) * (180/pi)
    if thetas == nothing
      thetas = theta
    else
      thetas = vcat(thetas, theta)
    end
    push!(thetaidx, thetaidx[end] + length(theta))
  end
  return thetas, thetaidx
end

function get_angledist(thetas, angle)
  diff = thetas - angle
  idx1 = diff .>=  180
  idx2 = diff .<= -180
  diff[idx1] = diff[idx1] - 360
  diff[idx2] = diff[idx2] + 360
  return abs(diff)
end

function get_pixelvals(thetas, angle; angle_threshold = 45)
  angle2 = (angle + 180) % 360
  angledist1 = get_angledist(thetas, angle)
  angledist2 = get_angledist(thetas, angle2)
  mindist = min(angledist1, angledist2)
  valid_idx = mindist .<= angle_threshold
  invalid_idx = mindist .> angle_threshold
  mindist[valid_idx] = 1 - (mindist[valid_idx] / angle_threshold)
  mindist[invalid_idx] = 0
  return mindist
end

function transform(points; newmax=23,  newmin=2)
  transpoints =  copy(points)
  newrange = newmax - newmin
  (xmax, ymax) = maximum(transpoints, 2)
  (xmin, ymin) = minimum(transpoints, 2)
  rangex = xmax - xmin
  rangey = ymax - ymin
  if rangex != 0
    transpoints[1, :] = (transpoints[1, :] - xmin)*newrange / rangex + newmin
  else
    transpoints[1, :] = fill(newmin, size(transpoints[1, :]))
  end
  if rangey != 0
    transpoints[2, :] = (transpoints[2, :] - ymin)*newrange / rangey + newmin
  else
    transpoints[2, :] = fill(newmin, size(transpoints[2, :]))
  end
  return floor(transpoints)
end

function points2im(points, end_indices, imsize, pixelvals, pixidx; endpoints = false)
  image = zeros(imsize)
  for strokenum=1:(length(end_indices)-1)
    start_ind = end_indices[strokenum]+1
    end_ind = end_indices[strokenum+1]
    stroke = points[1:2, start_ind:end_ind]
    spixels = pixelvals[pixidx[strokenum]+1 : pixidx[strokenum+1]]
    image = drawbresenham(stroke, spixels, image)
  end
  return image
end

function markendpoints(points, end_indices, imsize)
  image = zeros(imsize)
  for strokenum=1:(length(end_indices)-1)
    start_ind = end_indices[strokenum]+1
    end_ind = end_indices[strokenum+1]
    r_beg, c_beg = Int(round(points[2, start_ind])), Int(round(points[1, start_ind]))
    r_end, c_end = Int(round(points[2, end_ind])), Int(round(points[1, end_ind]))
    image[r_beg, c_beg] = 1
    image[r_end, c_end] = 1
  end
  return image
end

function drawbresenham(stroke, pixels, image)
  for i=1:length(pixels)
    x1, y1 =  stroke[1, i ], stroke[2, i ]
    x2, y2 = stroke[1, i+1], stroke[2, i+1]
    if pixels[i] > 0
      x, y = [ x1 x2 ] , [ y1 y2 ] #bresenham(x1, y1, x2, y2)
      for j = 1:length(x)
        r, c = Int(round(y[j])), Int(round(x[j]))
        if image[r, c] < pixels[i]
          image[r, c] = pixels[i]
        end
      end
    end
  end
  return image
end

function bresenham(x1, y1, x2, y2)
  x1 = Int(round(x1)); x2 = Int(round(x2));
  y1 = Int(round(y1)); y2 = Int(round(y2));
  dx=abs(x2-x1);
  dy=abs(y2-y1);
  steep=abs(dy)>abs(dx);
  if steep t=dx;dx=dy;dy=t; end

  #The main algorithm goes here.
  if dy==0
      q=zeros(1, dx+1)
  else
      q=[0 diff( mod( div(dx, 2):-dy:-dy*dx+div(dx, 2) , dx ) )  .>= 0]
      println("here")
  end
  println("size q = $(size(q))")
  #and ends here.

  if steep
    if y1<=y2
      y=collect(y1:y2)
    else
      y=collect(y1:-1:y2)
    end
    if x1<=x2
      x=x1+cumsum(q)
    else
      x=x1-cumsum(q)
    end
  else
    if x1<=x2
      x=collect(x1:x2);
    else
      x=collect(x1:-1:x2);
    end
    if y1<=y2
      y=y1+cumsum(q);
    else
      y=y1-cumsum(q);
    end
  end
  println("x size =$(size(x)) y size = $(size(y)) ")
  return x, y
end

function test()
  datapath = "../data/"
  filename = "full_simplified_airplane.ndjson"
  filepath = "$datapath$filename"
  sketches = get_sketch_objects(filepath)
  sketch = sketches[1]
#=  savesketch(sketch, "original.png")
  println("before resample sketch points dims = $(size(sketch.points))")
  sketch.points, sketch.end_indices = resample_sketch(sketch)
  println("after resampling sketch points dims = $(size(sketch.points))")
  savesketch(sketch, "resampled.png")
  sketch.points = normalize(sketch.points; d=2)
  @assert(std(sketch.points, 2)[1] == 1 && std(sketch.points, 2)[2] == 1)
  savesketch(sketch, "normalized.png")
  thetas, thetaidx = coords2angles(sketch.points, sketch.end_indices)
  pixelvals = get_pixelvals(thetas , 0)
  sketch.points = transform(sketch.points)
  savesketch(sketch, "transformed.png")
  imsize = (24,24)
  image = points2im(sketch.points, sketch.end_indices, imsize, pixelvals, thetaidx)
  fig = figure()
  #imsave("idm90.png", image , origin = "upper")
  imshow(image , origin = "upper")
  savefig("idm90.png") =#
  extract(sketch)
end

function extract(sketch)
  imsize = (24,24)
  avgim = zeros(imsize)
  angles = [0, 45, 90, 135]
  sketch.points, sketch.end_indices = resample_sketch(sketch)
  println("after resampling sketch points dims = $(size(sketch.points))")
  savesketch(sketch, "resampled.png")
  sketch.points = normalize(sketch.points; d=2)
  @assert(std(sketch.points, 2)[1] == 1 && std(sketch.points, 2)[2] == 1)
  savesketch(sketch, "normalized.png")
  thetas, thetaidx = coords2angles(sketch.points, sketch.end_indices)
  sketch.points = transform(sketch.points)
  fig = figure()
  image = markendpoints(sketch.points, sketch.end_indices, imsize)
  imshow(image , origin = "upper")
  savefig("idm_end.png")
  for angle in angles
    pixelvals = get_pixelvals(thetas , angle)
    image = points2im(sketch.points, sketch.end_indices, imsize, pixelvals, thetaidx)
    fig = figure()
    imshow(image , origin = "upper")
    savefig("idm$(angle).png")
    close()
    avgim += image
  end
  fig = figure()
  imshow(avgim , origin = "upper")
  savefig("idm.png")
  close()
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="A Visual Approach to Sketched Symbol Recognition. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--ratio"; arg_type=Float64; default=50.0; help="0.")
    ("--coeff"; arg_type=Float64; default=1.01; help="0.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  if o[:testmode]
    test()
  end
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "IDM.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end
