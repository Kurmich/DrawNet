module Segmenter
using PyPlot
using DrawNet


function getstrokelabels(model, strokes, seqlens; genmodel = nothing)
  batchsize = 1
  labels = Int[]
  for i = 1:length(strokes)
    stroke = strokes[i]
    len = seqlens[i]
  #  println(size(stroke))
    h = encode(genmodel, stroke, len, batchsize; dprob=0, attn=haskey(model, :attn))
    h = h*model[:w1][1] .+ model[:w1][2]
    ypred = h*model[:pred][1] .+ model[:pred][2]
  #  println(size(h), size(model[:pred][1]))
  #  println(Array(ypred))
    class = indmax(Array(ypred))
    push!(labels, class)
  end
  return labels
end

export getstrokelabels
end
