# U-matrix — ported from somoclu/src/uMatrix.cpp

function calculate_umatrix!(umatrix, codebook, nSomX::Int, nSomY::Int, nDimensions::Int,
                            rectangular::Bool, toroid::Bool)
    min_dist = 1.5f0
    @inbounds for som_y1 in 0:(nSomY - 1)
        for som_x1 in 0:(nSomX - 1)
            dist = 0.0f0
            nodes_number = 0
            neuron1 = som_y1 * nSomX + som_x1 + 1
            for som_y2 in 0:(nSomY - 1)
                for som_x2 in 0:(nSomX - 1)
                    som_x1 == som_x2 && som_y1 == som_y2 && continue
                    tmp = map_distance(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY,
                                       rectangular, toroid)
                    if tmp <= min_dist
                        nodes_number += 1
                        neuron2 = som_y2 * nSomX + som_x2 + 1
                        dist += euclidean_feature_distance(codebook, codebook, neuron1, neuron2,
                                                          nDimensions)
                    end
                end
            end
            umatrix[som_y1 * nSomX + som_x1 + 1] = dist / Float32(nodes_number)
        end
    end
    return umatrix
end
