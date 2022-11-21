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

    Universe(bodies...) = new{length(bodies)}(collect(bodies))

    Base.length(::Universe{N}) where N = N
end

function rk4_nbody_integrator(u::Universe{N}, iteration_callback::Function, T) where N
    δ = step(T)
    bds = u.Bodies
    tpos = Array{FVec}(undef, N, 2)

    function single_body2body_rk4(t, i1, i2)
        GravitConst = G * bds[i2].m
        z = (t, r, v) -> v
        zp = function (t, r, v)
                d = norm(r)
                (d == 0) && return ZFVec
                return GravitConst .* r ./ d ^ 3
             end

        r = bds[i2].r .- bds[i1].r
        v = bds[i1].v
        
        k1 = δ .* zp(t, r, v)
        l1 = δ .* z(t, r, v)

        tv = v .+ k1 ./ 2
        tr = r .+ l1 ./ 2
        nt = t + δ / 2

        k2 = δ .* zp(nt, tr, tv)
        l2 = δ .* z(nt, tr, tv)

        tr = r .+ l2 ./ 2
        tv = v .+ k2 ./ 2

        k3 = δ .* zp(nt, tr, tv)
        l3 = δ .* z(nt, tr, tv)

        tr = r .+ l3
        tv = v .+ k3
        nt = t + δ

        k4 = δ .* zp(nt, tr, tv)
        l4 = δ .* z(nt, tr, tv)
        
        dv = (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4) ./ 6
        dr = (l1 .+ 2 .* l2 .+ 2 .* l3 .+ l4) ./ 6

        return (dr, dv)
    end

    function single_body_rk4(t_0, i1)
        netr = bds[i1].r
        netv = bds[i1].v
        for i2 in 1:N
            (i2 == i1) && continue
            dr, dv = single_body2body_rk4(t_0, i1, i2)
            netr = netr .+ dr
            netv = netv .+ dv
        end
        return (netr, netv)
    end

    i = 1
    for t in T
        Threads.@threads for b in 1:N
            tpos[b, :] .= single_body_rk4(t, b)
        end

        for b in 1:N
            bds[b] = Body(bds[b].m, tpos[b, 1], tpos[b, 2])
        end

        iteration_callback(t, i)
        i += 1
    end

end

function simulate(T, earth, bodies...)
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
                pts[n - 1] = u.Bodies[n].r
            end
            
            frame(a, scatter(plt, pts, color = :jet))
            println("Finished Iteration $i")
        end, T)

    return gif(a)
end


earth = Body(20, [0, 0, 0], [0, 0, 0])
moon = Body(90, [50, 30, 50], [0, 0, 0])
simulate(0:.1:200, earth, moon)