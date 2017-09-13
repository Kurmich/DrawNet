global const cellpath = "../cells/"
function get_saturated_inds(seqcellvals; mincutoff = 0.1, maxcutoff = 0.9)
  mincells = trues(size(seqcellvals[1]))
  maxcells = trues(size(seqcellvals[1]))
  for cellvals in seqcellvals
    mincells &= (cellvals .<= mincutoff)
    maxcells &= (cellvals .>= maxcutoff)
  end
  #get valid indices
  mininds = find(mincells)
  maxinds = find(maxcells)
  return mininds, maxinds
end


function get_inds_in_range(seqcellvals; mincutoff = 0.1, maxcutoff = 0.9)
  cells = trues(size(seqcellvals[1]))
  for cellvals in seqcellvals
    cells &= ((cellvals .>= mincutoff) & (cellvals .<= maxcutoff))
  end
  #get valid indices
  inds = find(cells)
  return inds
end

function save_cell_images(cellinds, cellname, saturtype, sketch)
  path = "$(cellpath)/$(cellname)/$(saturtype)/"
  if !ispath(path)
    mkpath(path)
  end
  for cellind in cellinds
    p = string(path, cellind)
    if !ispath(p)
      mkpath(p)
    end
    imgcount = length(readdir(p))
    imgname = "$(p)/img$(imgcount+1).png"
    #info("saving sketch")
    savesketch(sketch, imgname; imsize=128)
  end
end

function save_saturated_inds(seqcellvals, cellname, sketch; mincutoff = 0.1, maxcutoff = 0.9)
  mininds, maxinds = get_saturated_inds(seqcellvals; mincutoff = mincutoff, maxcutoff = maxcutoff)
  save_cell_images(mininds, cellname, "min", sketch)
  save_cell_images(maxinds, cellname, "max", sketch)
end
