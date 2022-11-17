using Plots
using LinearAlgebra

const FP = Float64
const FVec = NTuple{3, FP}
const FZeroFvec = ntuple(x->FP(0), 3)
const G = 3

struct Body
    m::FP
    r::FVec
    v::FVec

    Body(m, r, v) = new(m, FVec(r), FVec(v))
end

struct Universe{N}
    Bodies::Vector{Body}
    TemporaryPositions::Array{FVec, 2}

    Universe(bodies...) = new{length(bodies)}(collect(bodies), Array{FVec}(undef, length(bodies), 2))

    Base.length(::Universe{N}) where N = N
end

function rk4_nbody_integrator(u::Universe{N}, iteration_callback::Function, t_0, t_f, δ = 1) where N
    f = (t, r, v, i1, i2) -> v
    g = (t, r, v, i1, i2) -> G .* r ./ norm(r) ^ 3

    if t_f < t_0
        iter = Int(round((t_0 - t_f) / δ))
        δ *= -1
    else
        iter = Int(round((t_f - t_0) / δ))
    end

    function single_body2body_rk4(t, i1, i2)
        r = u.Bodies[i2].r .- u.Bodies[i1].r
        v = u.Bodies[i1].v
        tv = r
        tr = v

        k1 = δ .* f(t, r, v, i1, i2)
        l1 = δ .* g(t, r, v, i1, i2)

        nt = t + δ / 2
        tr = r .+ k1 ./ 2
        tv = v .+ l1 ./ 2

        k2 = δ .* f(nt, tr, tv, i1, i2)
        l2 = δ .* g(nt, tr, tv, i1, i2)

        tr = r .+ k2 ./ 2
        tv = v .+ l2 ./ 2

        k3 = δ .* f(nt, tr, tv, i1, i2)
        l3 = δ .* g(nt, tr, tv, i1, i2)

        tr = r .+ k3
        tv = v .+ l3
        nt = t + δ

        k4 = δ .* f(nt, tr, tv, i1, i2)
        l4 = δ .* g(nt, tr, tv, i1, i2)
        
        dr = (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4) ./ 6
        dv = (l1 .+ 2 .* l2 .+ 2 .* l3 .+ l4) ./ 6

        return (dr, dv)
    end

    function single_body_rk4(t_0, i1)
        netr = u.Bodies[i1].r
        netv = u.Bodies[i1].v
        for i in 1:N
            (i == i1) && continue
            dr, dv = single_body2body_rk4(t_0, i1, i)
            netr = netr .+ dr
            netv = netv .+ dv
        end
        return (netr, netv)
    end

    t = t_0
    for i in 1:iter
        Threads.@threads for b in 1:N
            u.TemporaryPositions[b, :] .= single_body_rk4(t, b)
        end

        for b in 1:N
            u.Bodies[b] = Body(u.Bodies[b].m, u.TemporaryPositions[b, 1], u.TemporaryPositions[b, 2])
        end

        iteration_callback(t, i)
        t += δ
    end

end

function simulate(t_f, earth, bodies...)
    u = Universe(earth, bodies...)
    N = length(bodies) + 1
    a = Animation()
    plt = plot3d(N, xlim = (-500, 500), ylim = (-500, 500), zlim = (-500, 500), legend = false)
    pts = Array{FVec, 1}(undef, N)
    pts[1] = FZeroFvec

    rk4_nbody_integrator(u, 
        function(t, i)
            earthb = u.Bodies[1].r
        
            for n in 2:N
                #Center To Earth
                pts[n - 1] = u.Bodies[n].r .- earthb
            end
            
            frame(a, scatter(plt, pts, color = :jet))
            println("Finished Iteration $i")
        end, 0, t_f)

    return gif(a)
end


earth = Body(200, [0, 0, 0], [0, 0, 0])
moon = Body(9, [1, 6, 3], [0, 0, 0])
simulate(100, earth, moon)