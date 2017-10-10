#include("../utils/DataLoader.jl")
#include("../utils/Drawing.jl")
module IDM
using DataLoader, Drawing
using ArgParse, PyPlot, Images, JLD

type IdmTuple
  stroke_ims
  avg_im
end

function get_im_stds(idmtuples)
  datastroke = AbstractFloat[]
  dataim = AbstractFloat[]
  for idmtuple in idmtuples
    for strokeim in idmtuple.stroke_ims
      pushall!(datastroke, strokeim)
    end
    pushall!(dataim, idmtuple.avg_im)
  end
  return std(dataim), std(datastroke)
end

function pushall!(data, image)
  for i = 1:length(image)
    push!(data, image[i])
  end
end

function get_idm_batch(idmtuples, idx, params::Parameters)
  @assert(idx >= 0, "index must be nonnegative")
  @assert(idx < params.numbatches, "index must be less number of batches")
  start_ind = idx * params.batchsize
  indices = (start_ind + 1):(start_ind + params.batchsize)
  return idm_indices_to_batch(idmtuples, indices, params)
end

function idm_indices_to_batch(idmtuples, indices, params)
  batch = []
  hsize = 3
  sigma = 10
  avg_batch = nothing
  lin_strokes = []
  for i in indices
    lin_avg_img = idmtuples[i].avg_im
    cur_stroke_list = []
    for stroke_im in idmtuples[i].stroke_ims
      push!(cur_stroke_list, reshape(stroke_im, (1, length(stroke_im)) ) )
    end
    push!(lin_strokes, cur_stroke_list)
    lin_avg_img = reshape( lin_avg_img, (1, length(lin_avg_img)) )
    if avg_batch == nothing
      avg_batch = lin_avg_img
    else
      avg_batch = vcat(avg_batch, lin_avg_img)
    end
  end
  return avg_batch, lin_strokes
end

function normalize(points; d = 2)
  center = mean(points, d)
  points = points .- center
  if any(isnan, points)
    error("Nan found after zero mean")
  end
  stdv = std(points, d)
  if stdv[1] == 0 || stdv[2] == 0 || any(isnan, stdv)
    return points
  end
#  println("std = $(stdv)")
  return  points ./ stdv
end

function maxdist(points)
  #=find distance of furthest point from center of sketch=#
  center = mean(points, 2)
  diff = points .- center
  return maximum(sqrt(sumabs2(diff, 1)))
end

function resample(points, end_indices; coeff = 1.01, ratio = 50)
  interval = ( coeff * maxdist(points) ) / ratio #spatial samling interval
  end_idx = Int[]
  new_points = nothing
  #starting index of first stroke is 0+1=1
  push!(end_idx, 0)
  for strokenum=1:(length(end_indices)-1)
    start_ind = end_indices[strokenum]+1 #start idx of stroke
    end_ind = end_indices[strokenum+1] #end idx of stroke
    newstroke, numpoints = resample_stroke(points; coeff = coeff, ratio = ratio, interval=interval, indices = start_ind:end_ind)
    if new_points == nothing
      new_points = newstroke
    else
      new_points = hcat(new_points, newstroke)
    end
    push!(end_idx, end_idx[end] + numpoints)
  end
  return new_points, end_idx
end

function resample_stroke(points::Array; coeff = 1.01, ratio = 50, indices = 1:size(points,2), interval = nothing)
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
#  println("numpoints=$(numpoints) size newpoints = $(size(newpoints))")
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
  #=MAKE SOME CHANGES HERE
  if rangex != 0 && rangey != 0
    if rangex > rangey
      transpoints[2, :] *= rangey/rangex
    elseif rangey > rangex
      transpoints[1, :] *= rangex/rangey
    end
  end
  =#

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
  #  @printf("%g %g %g %g", points[2, start_ind],  points[1, start_ind], points[2, end_ind], points[1, end_ind] )
    r_beg, c_beg = round(Int, points[2, start_ind]), round(Int, points[1, start_ind])
    r_end, c_end = round(Int, points[2, end_ind]), round(Int, points[1, end_ind])
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
  #    println("here")
  end
#  println("size q = $(size(q))")
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
#  println("x size =$(size(x)) y size = $(size(y)) ")
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

function downsample(image)
  (nrows, ncols) = size(image)
  result = zeros(div(nrows, 2), div(ncols, 2))
  for i = 1:size(result, 1)
    for j = 1:size(result, 2)
      si = 2(i-1) + 1
      sj = 2(j-1) + 1
      result[i, j] = maximum(image[si:si+1, sj:sj+1])
    end
  end
  return result
end

function extract(points, end_indices)
  points, end_indices = resample_points()
end

function extract_avg_idm(points, end_indices; imlen::Int = 24, smooth::Bool = true)
  imsize = (imlen, imlen)
  #avgim = zeros(imsize)
  angles = [0, 45, 90, 135]
  hsize = 3
  sigma = 10
  gf = Kernel.gaussian([sigma, sigma], [hsize, hsize])
  if any(isnan, points)
    error("Nan found befrore resample")
  end
  points, end_indices = resample(points, end_indices)
  points = normalize(points; d=2)
  thetas, thetaidx = coords2angles(points, end_indices)
  points = transform(points; newmax = imlen-1)
  avgim = markendpoints(points, end_indices, imsize)
  avgim = smooth? imfilter(avgim, gf) : (avgim)
  for angle in angles
    pixelvals = get_pixelvals(thetas , angle)
    image = points2im(points, end_indices, imsize, pixelvals, thetaidx)
    image = smooth? imfilter(image, gf) : (image)
    avgim += image
  end
  avgim /= 5
  feat = reshape(avgim, (1, length(avgim)))
  return avgim
end

function get_stroke_images(points, end_indices; imlen::Int = 12, smooth::Bool = true)
  strokeims = Any[]
  for strokenum=1:(length(end_indices)-1)
    start_ind = end_indices[strokenum]+1
    end_ind = end_indices[strokenum+1]
    stroke_end_indices = [0 (end_ind-start_ind+1)]
    image =  extract_avg_idm(points[:, start_ind:end_ind],stroke_end_indices; imlen = imlen, smooth = smooth)
    push!(strokeims, image)
  end
  return strokeims
end

function get_idm_objects(sketches; imlen::Int = 12, smooth::Bool = true)
  idmobjs = IdmTuple[]
  for sketch in sketches
    stroke_ims = get_stroke_images(sketch.points, sketch.end_indices; imlen = imlen, smooth=smooth)
    avgim = extract_avg_idm(sketch.points, sketch.end_indices; imlen = imlen, smooth=smooth)
    push!(idmobjs, IdmTuple(stroke_ims, avgim))
  end
  return idmobjs
end

function extractidm(points, end_indices)
  imsize = (24,24)
  hsize = 3
  sigma = 10
  avgim = zeros((12,12))
  angles = [0, 45, 90, 135]
  points, end_indices = resample(points, end_indices)
  points = normalize(points; d=2)
  thetas, thetaidx = coords2angles(points, end_indices)
  points = transform(points)
  gf = Kernel.gaussian([sigma, sigma], [hsize, hsize])
  endpnts = markendpoints(points, end_indices, imsize)
  endpnts = imfilter(endpnts, gf)
  endpnts = downsample(endpnts)
  feat = reshape(endpnts, (1, length(endpnts)))
  for angle in angles
    pixelvals = get_pixelvals(thetas , angle)
    image = points2im(points, end_indices, imsize, pixelvals, thetaidx)
    image = imfilter(image, gf)
    image = downsample(image)
    feat = hcat(feat, reshape(image, (1, length(image))))
  end
  return feat
end


function extract(sketch)
  return extractidm(sketch.points, sketch.end_indices)
  imsize = (24,24)
  hsize = 3
  sigma = 10
  avgim = zeros((12,12))
  angles = [0, 45, 90, 135]
  sketch.points, sketch.end_indices = resample(sketch.points, sketch.end_indices)
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
  gf = Kernel.gaussian([sigma, sigma], [hsize, hsize])
  for angle in angles
    pixelvals = get_pixelvals(thetas , angle)
    image = points2im(sketch.points, sketch.end_indices, imsize, pixelvals, thetaidx)
    #image = imfilter(image, gf)
    image = downsample(image)
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


function saveidm(im, name)
  fig = figure()
  imshow(im , origin = "upper")
  savefig(name)
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
    ("--datapath"; arg_type=String; default="../data/"; help="Number of epochs per checkpoint creation.")
    ("--filename"; arg_type=String; default="full_simplified_airplane.ndjson"; help="Decoder: lstm, or ....")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  if o[:testmode]
    test()
  end
  filepath = "$(o[:datapath])$(o[:filename])"
  info("getting sketch objects")
  sketches = get_sketch_objects(filepath)
  info("now getting idm objects")
  idmobjs = get_idm_objects(sketches)
  info("idm objects were extracted successfully")
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "IDM.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

export get_im_stds, get_idm_objects, get_idm_batch
export IdmTuple, save_idmtuples, load_idmtuples
export idm_indices_to_batch, saveidm, extractidm
end
