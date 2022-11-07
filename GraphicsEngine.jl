using LinearAlgebra
using Reel

const Point = NTuple{3, Float64}
const Vec = NTuple{3, Float64}

mutable struct ViewingPlane
    observer::Point
    unormal::Vec
    uplaneSide1::Vec
    uplaneSide2::Vec
    bounds::Tuple{2, Float64}
end

struct PointCollection
    collecter::Function
    len::Int

    PointCollection(collector, len) = new(collector, len)

    Base.length(pc::PointCollection) = pc.len
    Base.getindex(pc::PointCollection, i) = pc.collecter(i)
end

function pointdistance(vp::ViewingPlane, point::Point)
    viewVec::Point = vp.observer .- point
    dist = dot(viewVec, vp.unormal)
    isViewable = dist <= 0
    if isViewable
        pointVecInDirectionOfViewingPlane = dist * vp.unormal + point
        (isViewable & normalize(dist - vp.observer) <= vp.bound) && (return dist)
        return -1
    end
    return -1
end

function generate2DProjection(observer_vector, image_bounds, pointcollection)
    plane_normal_vector = normalize(observer_vector)
    plane_point = -observer_vector

    x_direction_vector = copy(plane_point)
    x_direction_vector[1] += image_bounds[1]
    normalize(x_direction_vector)

    y_direction_vector = copy(plane_point)
    y_direction_vector[2] += image_length[2]
    normalize(y_direction_vector)

    vp = ViewingPlane(plane_point, plane_normal_vector, x_direction_vector, y_direction_vector, image_bounds)
    mat = zeros(Float64, image_bounds[1], image_bounds[2])
    for i in 1:length(pointcollection)

    end
end

