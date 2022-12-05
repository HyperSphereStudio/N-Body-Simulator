module BlackHoleSimulator

using LinearAlgebra
using Plots

include("GeneralizedContextOperations.jl")
@CreateExecutionContext(false)
    
#Universal Gravitational Constant in km^2
#const G = 6.6743E-11 / 1000^2
const G = 7
const ZFVec = ntuple(x -> Float32(0), 3)
const FVec = NTuple{3, Float32}

struct Body
    m::Float32
    r::FVec
    v::FVec

    Body(m, r, v) = new(Float32(m), FVec(v), FVec(r))
end

mass(massi, idx) = massi[idx]
mass(massi, idx, m) = massi[idx] = m
pos(spatiali, idx) = spatiali[idx, 1]
pos(spatiali, idx, v) = spatiali[idx, 1] = v
vel(spatiali, idx) = spatiali[idx, 2]
vel(spatiali, idx, v) = spatiali[idx, 2] = v
tpos(spatiali, idx) = spatiali[idx, 3]
tpos(spatiali, idx, v) = spatiali[idx, 3] = v
tvel(spatiali, idx) = spatiali[idx, 4]
tvel(spatiali, idx, v) = spatiali[idx, 4] = v

function state_instance(s, m, ΔT)
    @iterate(function gravitational_kernel(b, s, m, ΔT)
                #Zero out Net Acceleration
                nacc = ZFVec

                #Net Acceleration
                for b2 in 1:N
                    r = pos(s, b2) .- pos(s, b)
                    d = norm(r)
                    F_g = Float32(d == 0 ? 0.0 : G * mass(m, b2) ./ d^3)
                    nacc = nacc .+ F_g .* r
                end
                
                tvel(s, vel(s, b) .+ nacc .* ΔT)
                tpos(s, pos(s, b) .+ tvel(s, b) .* ΔT)

            end, 1:N, s, m, ΔT)

    @iterate(function positional_copy_kernel(b, s)
                #Move temp to actual
                vel(s, b, tvel(s, b))
                pos(s, b, tpos(s, b))
            end, 1:N, s)
end

function simulate(spatial, mass, T, state_visitor::Function)
    i = 1
    for t in T
        state_instance(spatial, mass, step(T))
        state_visitor(spatial, mass, t, i)
        i += 1
    end
end

function nbodysim(T, bodies::Body...)
    N = length(bodies)
    a = Animation()

    spatial_information = TArray{FVec, 2}(undef, length(bodies), 4)
    mass_information = TArray{FP}(undef, length(bodies))
    for i in eachindex(bodies)
        spatial_information[i, 1] = bodies[i].r
        spatial_information[i, 2] = bodies[i].v
        mass_information[i] = bodies[i].m
    end

    println("Starting Universe! of $N bodies and $(length(T)) iterations")

    plt = plot3d(N, legend = false)
    pts = Array{FVec, 1}(undef, N)

    simulate(spatial_information, mass_information, T, 
            function (r, t, i)
                if i % 2 == 0
                    for n in 1:N
                        pts[n] = r[:pos, n]
                    end
                    frame(a, scatter(plt, pts, color = :jet))
                    println("Finished Iteration $i")
                end
        end)

    return display(gif(a))
end

blackhole = Body(4E2, [0, 0, 0], [0, 0, 0])
bodies = [Body(rand(10:1:40), rand(-5.0:1.0:5.0, 3), rand(0:0, 3)) for j in 1:1]
nbodysim(0.0:1.0:200.0, blackhole, bodies...)
end