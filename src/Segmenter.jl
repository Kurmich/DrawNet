module Segmenter
using PyPlot, Knet
using DrawNet


function getstrokelabels(model, strokes, seqlens; genmodel = nothing)
  batchsize = 1
  labels = Int[]
  for i = 1:length(strokes)
    stroke = strokes[i]
    len = seqlens[i]
    #encode vector using generative model
    h = encode(genmodel, stroke, len, batchsize; dprob=0, attn=haskey(model, :attn))
    h = relu(h*model[:w1][1] .+ model[:w1][2])
    h = relu(h*model[:w2][1] .+ model[:w2][2])
    ypred = h*model[:pred][1] .+ model[:pred][2]
    #find predicted class
    class = indmax(Array(ypred))
    push!(labels, class)
  end
  return labels
end

function getbatchstrokelabels()

end

export getstrokelabels
end
