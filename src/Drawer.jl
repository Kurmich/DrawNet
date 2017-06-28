include("DrawNet.jl")
module Drawer
using DrawNet, Distributions
function points_to_lines(points)
  strokes = []
  x_points = []
  y_points = []
  x = 0
  y = 0
  for i=1:size(points, 2)
    if points[3, i] == 1
      x += Float64(points[1, i])
      y += Float64(points[2, i])
      push!(x_points, x)
      push!(y_points, y)
      push!(strokes, vcat(x_points', y_points'))
      x_points = []
      y_points = []
    else
      x += Float64(points[1, i])
      y += Float64(points[2, i])
      push!(x_points, x)
      push!(y_points, y)
    end
  end
  return strokes
end

function adjust_temp(pipdf, temp)
  #=Assumption: pipdf is normalized=#
  return pipdf ./ temp
end

function get_pi_idx(x, pdf; temp=1.0 greedy::Bool = false)
  if greedy
    maxval, maxind = findmax(pdf)
    return maxind
  end
  pdf = adjust_temp(copy(pdf), temp)
  accumulate = 0
  for i=1:length(pdf)
    accumulate += pdf[i]
    if accumulate >= x
      return i
    end
  end
  info("Error with sampling ensemble")
  return -1
end

function sample_gaussian_2d(mu1, mu2, s1, s2, rho; temp = 1.0, greedy::Bool = false)
  if greedy
    return mu1, mu2
  end
  mean = [mu1, mu2]
  s1 *= temp*temp
  s2 *= temp*temp
  cov = [ s1*s1  rho*s1*s2; rho*s1*s2 s2*s2 ]
  x, y = rand(MvNormal(mean,cov)) #sample from multivariate normal
  return x, y
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="Sketch sampler from model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--modelname"; arg_type=String; default="full_simplified_airplane.jld"; help="Name of the pretrained model")
    ("--datasetname"; arg_type=Int; default=2048; help="Name of the dataset")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  w = load(o[:modelname])
  model = revconvertmodel(w["model"])
  trnpoints3D, vldpoints3D, tstpoints3D = loaddata(sketchpoints3D)

end
end
