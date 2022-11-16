#Array 1 (Rows = Object Count, Depth = 2 [r, v], Columns = 3)
#Array 2 (Columns = Mass of Object)

const FP = Float64

struct Body
    m
    r
    v

    Body(m, r, v) = new(m, r, v)
end

mutable struct Universe
    SpacialInfo::Array{FP, 3}
    MassInfo::Array{FP, 1}

    function Universe(bodies...)
        SpacialInfo = Array{FP, 3}(undef, length(bodies), 2, 3)
        MassInfo = Array{FP, 1}(undef, length(bodies))

        for i in eachindex(bodies)
            SpacialInfo[i, 1, :] = bodies[i].r
            SpacialInfo[i, 2, :] = bodies[i].v
            MassInfo[i] = bodies[i].m
        end

        new(SpacialInfo, MassInfo)
    end

    Base.length(u::Universe) = length(u.MassInfo)
end

b1 = Body(4, [2, 1, 3], [65, 1, 7])
b2 = Body(1, [-2, 1, -3], [-65, 1, -7])
u = Universe(b1, b2)

#Acceleration of Body 2 on Body 1
function ForceOfGravity(u::Universe, b1, b2)
    r = u.SpacialInfo[b2, 1, :] - u.SpacialInfo[b1, 1, :]
    R = norm(r)
    return G * u.MassInfo[b2] * r / R ^ 3
end

function Acceleration(u::Universe, b1)
    NetAcceleration = zeros(FP, 3)

    for b in 1:length(u)
        (b == b1) && continue
        NetAcceleration += ForceOfGravity(u, b1, b)
    end

    
end

#∑(a) = ∑(-G * M * r / r^3)










