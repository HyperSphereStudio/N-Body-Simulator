using ProgressMeter
using GeometryBasics
using GLMakie
using FileIO
using Printf
using StaticArrays

const FP = Float64
const G = 6.67259E-20
const BodySizeScale = 1E3
const EnableMultiThreading = false
const Limit = 1.5E4
const UseDynamicLimits = false
const FPS = 20

struct Body
    m::FP
    pv::Array{FP, 2}
    tpv::Array{FP, 2}
    planet_size::Int32

    function Body(m, r, v, s)
        pv = zeros(FP, 2, 3)
        pv[1, :] = r
        pv[2, :] = v
        return new(m, pv, zeros(FP, 2, 3), s * BodySizeScale)
    end
end

struct Universe{N}
    Bodies::Vector{Body}
    Universe(bodies...) = new{length(bodies)}(collect(bodies))
    Base.length(::Universe{N}) where N = N
end

#Integrators
function rk4_integrator(t, δ, pv, va)
    k1 = δ .* va(t, pv)               
    k2 = δ .* va(t + δ/2, pv .+ k1 ./ 2)
    k3 = δ .* va(t + δ/2, pv .+ k2 ./ 2)
    k4 = δ .* va(t + δ, pv .+ k3)
    return pv .+ (k1 + 2 * k2 + 2 * k3 + k4) / 6
end
polynomial_integrator(t, δ, pv, va) = pv .+ δ .* va(t, pv)

rows(x) = size(x, 1)
cols(x) = size(x, 2)


#Helper function to annotate concurrent capability
macro MultiThread(ex)
    return EnableMultiThreading ? esc(:(Threads.@threads $ex)) : esc(ex)
end

function nbody(u::Universe{N}, integrator::F, iteration_callback::F2, T) where {N, F, F2}
    δ = step(T)
    bds = u.Bodies

    i = 1
    for t in T
        @MultiThread for b in 1:N
            bod = u.Bodies[b]

            #Set the New PV to the temporary PV so that it does not effect concurrent operations
            bod.tpv .= integrator(t, δ, bod.pv,   
                function single_nbody_velacc(t, pv)
                    va = zeros(FP, 2, 3)
                    a = @view(va[2, :])                     #View the Acceleration Vector from the VA Matrix
                    p = @view(pv[1, :])                     #View the Position Vector from the PV Matrix
                    va[1, :] .= pv[2, :]                    #Because z = v, pass the current velocity as next velocity
                    for b2 in 1:N
                        q = bds[b2]
                        r = @view(q.pv[1, :]) - p           #Solve for the r vector representing the vector from p -> q 
                        d = sqrt(sum(r -> r ^ 2, r))        #Solve for the distance of r
                        (b == b2 || d == 0) && continue     #Probaly the same object so have it stick or do nothing
                        a .+= G * r * q.m / d ^ 3
                    end
                    return va
                end)
        end

        #Copy the Temporary Positon to the Real PV
        for b in u.Bodies
            b.pv .= b.tpv
        end
        iteration_callback(t, i)
        i += 1
    end
end

#Creates Progress Bar
function Progressable(x::Function, ticks, desc, color)
    prog = Progress(ticks, desc=desc, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=color)
    update!(prog, 1)
    x(prog)
end


#sum(m_i * p_i) / sum(m_i) = com
function CenterOfMass(bodies)
    com = zeros(FP, 3)
    m = 0.0
    for b in bodies
        com .+= b.m * @view(b.pv[1, :])
        m += b.m
    end
    com ./= m
    return com
end

GenerateLimits(pos) = ntuple(i -> (FP(pos[i] - Limit), FP(pos[i] + Limit)), 3)

BuildPlot(bodies, snap_shot, N) = Progressable(rows(snap_shot), "Building Plot", :blue) do prog
    images = load.(["Assets/Earth.png", "Assets/FAROOQM.jpg", "Assets/Moon.jpg"])
    set_theme!(backgroundcolor = :black)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect = (1, 1, 1), viewmode=:fit, limits = GenerateLimits([0, 0, 0]))

    ax.titlecolor = :red
    ax.xgridcolor = :green
    ax.ygridcolor = :green
    ax.zgridcolor = :green
    ax.xticklabelcolor = :green
    ax.yticklabelcolor = :green
    ax.zticklabelcolor = :green
    ax.title = "$N Body Solution"

    #Initialize Observables
    meshes = Array{Any}(undef, N)
    trajectory_positions = [Observable(Point3f[]) for i in 1:(N+1)]

    for i in 1:N
        #Initialize Meshes
        meshes[i] = mesh!(ax, Sphere(Point3f(0), bodies[i].planet_size), color = images[i > 3 ? 3 : i])
        
        for b in 1:N; lines!(ax, trajectory_positions[b], color = :gray) end     #Initialize Trajectories
        lines!(ax, trajectory_positions[end], color = :red)                      #Initialize COM Trajectory
    end

    return record(fig, "Trajectory.gif", 1:rows(snap_shot); framerate = FPS) do f
        for w in 1:N
            translate!(meshes[w], snap_shot[f, w])
        end

        for w in 1:(N + 1)
            push!(trajectory_positions[w][], snap_shot[f, w])
            notify(trajectory_positions[w])
        end

        if UseDynamicLimits
            autolimits!(ax)
        else
            ax.limits[] = GenerateLimits(snap_shot[f, end])
        end
        update!(prog, f)
    end
end


#Run this function on every frame to capture the state of the system
function CapturePoints(body_pos, snap_shot, com, shap_shot_idx)
    for i in 1:length(body_pos)
        r = body_pos[i]
        snap_shot[shap_shot_idx, i] = (r[1], r[2], r[3])
    end
    snap_shot[shap_shot_idx, end] = (com[1], com[2], com[3])
end

InitWriteDataFile(N) = begin
    io = open("data.dat", "w")
    write(io, "t\t\t|")
    for i in 1:N
        write(io, "\t\t\t\tr$i\t\t\t|\t\t\tv$i\t\t\t\t|")
    end
    write(io, "\n")
    return io
end

numf(n) = string(@sprintf("%10.2f", n))

WriteDataFileEntry(bodies, t, io) = begin
    write_array(v) = join([numf(i) for i in v])
    write(io, "$(numf(t))\t|")
    for b in bodies
        write(io, "\t$(write_array(@view(b.pv[1, :])))\t|\t$(write_array(@view(b.pv[2, :])))\t|")
    end
    write(io, "\n")  
end

function simulate(T, integrator, bodies...; plotevery = 50, writedata = false)
    u = Universe(bodies...)
    N = length(bodies)
    body_pos = [view(b.pv, 1, 1:3) for b in u.Bodies]
    shap_shot_idx = 1
    frames = Int(floor(length(T) / plotevery))
    snap_shot = Array{NTuple{3, FP}, 2}(undef, frames, N + 1)

    println("Initializing $N body Universe")
    println("T_0 = $(first(T)) seconds")
    println("Step = $(step(T)) seconds")
    println("T_F = $(last(T)) seconds")

    if writedata
        io = InitWriteDataFile(N)
        WriteDataFileEntry(u.Bodies, 0, io)
    end
    
    function print_state_vectors(isFirst)
        name = isFirst ? "i" : "f"
        for i in 1:N
            isFirst && (println("m_$i = ", round(u.Bodies[i].m, digits=2)))
            println("R_{$name, $i} = ", round.(u.Bodies[i].pv[1, :], digits=2), " km")
            println("V_{$name, $i} = ", round.(u.Bodies[i].pv[2, :], digits=2), " km")
            println()
        end
    end

    print_state_vectors(true)

    Progressable(frames, "Performing Universe Calculations", :yellow) do prog
        nbody(u, integrator,
            function(t, i)
                if i % plotevery == 0
                    CapturePoints(body_pos, snap_shot, CenterOfMass(u.Bodies), shap_shot_idx)
                    update!(prog, shap_shot_idx)
                    writedata && WriteDataFileEntry(u.Bodies, t, io)
                    shap_shot_idx += 1
                end
            end, T)
    end

    print_state_vectors(false)
    writedata && (close(io))

    return BuildPlot(u.Bodies, snap_shot, N)
end

earth = Body(1E26, [0, 0, 0], [15, 25, 35], 2)
moon = Body(1E25, [3000, 2000, 0], [-5, 10, -10], 1)
moon2 = Body(1E25, [3000, -2000, 0], [-5, -10, 10], 1)
moon3 = Body(1E25, [-3000, -2000, 0], [-5, 10, 10], 1)

simulate(0.0:.1:3600, rk4_integrator, earth, moon, moon2, moon3; plotevery=100, writedata = true)