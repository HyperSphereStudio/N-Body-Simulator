using Plots
using ProgressMeter
gr()

const FP = Float64
const G = 6.67259E-20

struct Body
    μ::FP
    pv::Array{FP, 2}
    tpv::Array{FP, 2}

    function Body(m, r, v)
        pv = zeros(FP, 2, 3)
        pv[1, :] = r
        pv[2, :] = v
        return new(G * m, pv, zeros(FP, 2, 3))
    end
end

struct Universe{N}
    Bodies::Vector{Body}
    Universe(bodies...) = new{length(bodies)}(collect(bodies))
    Base.length(::Universe{N}) where N = N
end

function rk4_integrator(t, δ, pv, va)
    k1 = δ .* va(t, pv)               
    k2 = δ .* va(t + δ/2, pv .+ k1 ./ 2)
    k3 = δ .* va(t + δ/2, pv .+ k2 ./ 2)
    k4 = δ .* va(t + δ, pv .+ k3)
    return pv .+ (k1 + 2 * k2 + 2 * k3 + k4) / 6
end

polynomial_integrator(t, δ, pv, va) = pv .+ δ .* va(t, pv)

function nbody(u::Universe{N}, integrator::F, iteration_callback::F2, T) where {N, F, F2}
    δ = step(T)
    bds = u.Bodies

    i = 1
    for t in T
        for b in 1:N
            bod = u.Bodies[b]

            bod.tpv .= integrator(t, δ, bod.pv,   
            function single_nbody_velacc(t, pv)
                va = zeros(FP, 2, 3)
                a = @view(va[2, :])
                p = @view(pv[1, :])
                va[1, :] .= pv[2, :]
                for b2 in 1:N
                    q = bds[b2]
                    r = @view(q.pv[1, :]) - p                        #Solve for the r vector representing the vector from p -> q 
                    d = sqrt(sum(r -> r ^ 2, r))                     #Solve for the distance of r
                    (b == b2 || d == 0) && continue                  #Probaly the same object so have it stick or do nothing
                    a .+= r * q.μ / d ^ 3
                end
                return va
            end)
        end

        for b in 1:N
            bod = u.Bodies[b]
            bod.pv .= bod.tpv
        end
        iteration_callback(t, i)
        i += 1
    end
end

function simulate(T, integrator, bodies...; plotevery = 50)
    u = Universe(bodies...)
    N = length(bodies)
    pts = [view(b.pv, 1, 1:3) for b in u.Bodies]
    com = zeros(FP, 3)
    fei = 1
    numPts = Int(floor(length(T) / plotevery))
    points = Array{NTuple{3, FP}, 2}(undef, numPts, N + 1)

    function CapturePoints()
        for i in 1:length(pts)
            r = pts[i]
            points[fei, i] = (r[1], r[2], r[3])
        end
        points[fei, end] = (com[1], com[2], com[3])
    end

    function CenterOfMass()
        com .= 0
        m = 0.0
        for i in 1:N
            com .+= u.Bodies[i].μ * @view(u.Bodies[i].pv[1, :])
            m += u.Bodies[i].μ
        end
        com ./= m
        return com
    end

    function BuildPlot()
        prog = Progress(numPts, desc="Building Plot", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:blue)
        plt = plot3d(N + 1, 
            xlabel="X [km]", ylabel="Y [km]", zlabel = "Z [km]", title = "$N Body Space Curve", 
            xlims=(-1E5, 1E5), ylims=(-1E5, 1E5), zlims=(-1E5, 1E5),
            legend = false)
        a = Animation()
        trajectories = [Vector{FP}(undef, fei * (N + 1)) for i in 1:3]
        
        r = 1
        for b in 1:(N+1)
            for i in 1:3
                trajectories[i][r] = points[r, b][i]
            end
            r += 1
        end

        pei = 1
        for r in eachrow(points)
            scplt = scatter(plt, r, markersize = 3, color=:red)
          #  scatter!(trajectories, markersize = 1, color=:gray)
            frame(a, scplt)
            update!(prog, pei)
            pei += 1
        end
        return gif(a, fps=5)
    end
    
    prog = Progress(numPts, desc="Performing Universe Calculations", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    nbody(u, integrator,
        function(t, i)
            if i % plotevery == 0
                CenterOfMass()
                CapturePoints()
                update!(prog, fei)
                fei += 1
            end
        end, T)
    
    println([b.pv for b in u.Bodies])

    println("Universe Calculations completed")   
    return BuildPlot()
end

earth = Body(1E26, [0, 0, 0], [15, 25, 35])
moon = Body(1E25, [3000, 2000, 0], [-5, 10, -10])
moon2 = Body(1E25, [3000, -2000, 0], [-5, -10, 10])

simulate(0.0:.5:3600, rk4_integrator, earth, moon, moon2; plotevery=100)