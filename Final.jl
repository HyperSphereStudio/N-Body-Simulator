using Plots
using FastBroadcast
gr()

const FP = Float64
const G = 6.68E-20

struct Body
    μ::FP
    pv::Array{FP, 2}
    tpva::Array{FP, 2}
    tlk::Array{FP, 2}
    tnk::Array{FP, 2}

    function Body(m, r, v)
        pv = zeros(FP, 2, 3)
        pv[1, :] = r
        pv[2, :] = v
        return new(G * m, pv, zeros(FP, 3, 3), zeros(FP, 2, 3), zeros(FP, 2, 3))
    end
end

struct Universe{N}
    Bodies::Vector{Body}
    Universe(bodies...) = new{length(bodies)}(collect(bodies))
    Base.length(::Universe{N}) where N = N
end

function rk4(t, δ, pv, tpv, lk, nk, va)
    tpv .= pv
    lk .= δ .* va(t, pv)
    nk .= lk

    tpv .= pv .+ lk ./ 2
    lk .= δ .* va(t + δ/2, tpv)
    nk .+= 2 .* lk

    tpv .= pv .+ lk ./ 2
    lk .= δ .* va(t + δ/2, tpv)
    nk .+= 2 .* lk

    tpv .= pv .+ lk
    lk .= δ .* va(t + δ, tpv)
    nk .+= lk

    tpv .= pv .+ nk / 6
end

polynomial_integrator(t, δ, pv, tpv, lk, nk, va) = tpv .= pv .+ δ .* va(t, pv)

function nbody(u::Universe{N}, integrator::F, iteration_callback::F2, T) where {N, F, F2}
    δ = step(T)
    bds = u.Bodies

    i = 1
    for t in T
        Threads.@threads for b in 1:N
            bod = u.Bodies[b]
            tpva = bod.tpva

            r = @view(tpva[1, :])
            tpv = @view(tpva[1:2, :])
            va = @view(tpva[2:3, :])
            v = @view(tpva[2, :])
            a = @view(tpva[3, :])

            integrator(t, δ, bod.pv, tpv, bod.tlk, bod.tnk,   
                function single_nbody_velacc(t, pv)
                    a .= 0
                    for b2 in 1:N
                        q = bds[b2]
                        r .= @view(q.pv[1, :]) .- @view(pv[1, :])           #Solve for the r vector representing the vector from p -> q 
                        d = sqrt(sum(r -> r ^ 2, r))                        #Solve for the distance of r
                        (d == 0) && continue                                #Probaly the same object so have it stick or do nothing
                        a .+= r .* q.μ / d ^ 3
                    end
                    va[1, :] .= pv[2, :]
                    return va
                end)
        end

        for b in 1:N
            bod = u.Bodies[b]
            bod.pv .= @view(bod.tpva[1:2, :])
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
    points = Array{NTuple{3, FP}, 2}(undef, Int(floor(length(T) / plotevery)), N + 1)

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
        println("Building Plot...")
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

        for r in eachrow(points)
            scplt = scatter(plt, r, markersize = 3, color=:red)
            scatter!(trajectories, markersize = 1, color=:gray)
            frame(a, scplt)
        end
        return gif(a, fps=10)
    end
    
    nbody(u, integrator,
        function(t, i)
            if i % plotevery == 0
                CenterOfMass()
                CapturePoints()
                println("Finished Iteration $i")
                fei += 1
            end
        end, T)

    println("Universe Calculations completed")   
    return BuildPlot()
end

earth = Body(1E26, [0, 0, 0], [15, 25, 35])
moon = Body(1E25, [3000, 2000, 0], [-5, 10, -10])
#moon2 = Body(1E-5, [0, -100, 0], [0, 0, -sqrt(G)])
simulate(0.0:.01:3600.0, polynomial_integrator, earth, moon; plotevery=1000)