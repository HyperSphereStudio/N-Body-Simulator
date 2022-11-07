using StaticArrays

struct TRef{T}
    x::Ptr{UInt8}

    TRef{T}(x::Ptr{UInt8}) where T = new{T}(x)
    TRef{T}(x::Ptr{Cvoid}) where T = new{T}(Base.unsafe_convert(Ptr{UInt8}, x))
    TRef{T}(x::Ptr{T}) where T = TRef{T}(Base.unsafe_convert(Ptr{UInt8}, x))
    TRef{T}(x::Ref{T}) where T = TRef{T}(pointer_from_objref(x))

    @inline Base.pointer(x::TRef) = getfield(x, :x)
    @inline Base.getproperty(x::TRef, s::Symbol) = getproperty(x, Val{s}())
    @inline Base.setproperty!(x::TRef, s::Symbol, v) = setproperty!(x, Val{s}(), v)
    @inline Base.getproperty(::TRef{T}, ::Val{N}) where {N, T} = throw("type $T has no field $N")
    @inline Base.setproperty!(::TRef{T}, ::Val{N}, v) where {N, T} = throw("type $T has no field $N")
    @inline Base.getindex(x::TRef{T}) where T = Base.unsafe_load(Base.unsafe_convert(Ptr{T}, pointer(x)))
end

struct TVal{T, P}
    v::P

    TVal{T, P}(x::T) where {T, P} = new{T, P}(reinterpret(P, x))
    @inline Base.pointer(x::TVal) = pointer(x.v)
    @inline Base.getproperty(x::TVal, s::Symbol) = getproperty(x, Val{s}())
    @inline Base.setproperty!(x::TVal, s::Symbol, v) = setproperty!(x, Val{s}(), v)
    @inline Base.getproperty(::TVal{T, P}, ::Val{N}) where {N, T, P} = throw("type $T has no field $N")
    @inline Base.setproperty!(::TVal{T, P}, ::Val{N}, v) where {N, T, P} = throw("type $T has no field $N")
end

struct GClass{R, V, B, U} 
    GClass{R, V, B, U}() where {R, V, B, U} = new{R, V, B, U}()
    function Base.iterate(g::GClass, s=1)
        if s == 1
            return (RefType(g), 2)
        elseif s == 2
            return (ValType(g), 3)
        elseif s == 3
            return (ValType(g), 4)
        elseif s == 4
            return (SharedType(g), nothing)
        else
            return nothing
        end
    end
end

RefType(::GClass{R, V, B, U}) where {R, V, B, U} = R
ValType(::GClass{R, V, B, U}) where {R, V, B, U} = V
BaseType(::GClass{R, V, B, U}) where {R, V, B, U} = B
SharedType(::GClass{R, V, B, U}) where {R, V, B, U} = U

const GClasses = Dict{Type, GClass}()

function GClass(ty::Type)
    haskey(GClasses, ty) && return GClasses[ty]
    eval(x) = Base.eval(x)

    rty = TRef{ty}
    vty = TVal{ty, SVector{sizeof(ty), UInt8}}
    uty = Union{rty, vty}
    GClasses[ty] = GClass{rty, vty, ty, Union{ty, rty, vty}}()

    for i in 1:fieldcount(ty)
        name = fieldname(ty, i)
        offset = fieldoffset(ty, i)
        fty = fieldtype(ty, i)
        flen = sizeof(fty)
        
        if fieldcount(fty) !== 0
            fclz = GClass(fty)
            frclz = RefType(fclz)
            eval(quote 
                    @inline Base.getproperty(gsr::$rty, ::Val{$(QuoteNode(name))}) = $rty(pointer(gsr) + $offset)
                    @inline Base.setproperty!(gsr::$rty, w::Val{$(QuoteNode(name))}, v::$fty) = unsafe_store!(Base.unsafe_convert(Ptr{$fty}, pointer(gsr) + $offset), v)
                    @inline Base.setproperty!(gsr::$rty, ::Val{$(QuoteNode(name))}, v::$frclz) = unsafe_store!(Base.unsafe_convert(Ptr{$fty}, pointer(gsr) + $offset), v[])
                end)
        else
            eval(quote 
                    @inline Base.getproperty(gsr::T, ::Val{$(QuoteNode(name))}) where T <: $uty = unsafe_load(Base.unsafe_convert(Ptr{$fty}, pointer(gsr) + $offset))
                    @inline Base.setproperty!(gsr::T, ::Val{$(QuoteNode(name))}, v::$fty)  where T <: $uty = unsafe_store!(Base.unsafe_convert(Ptr{$fty}, pointer(gsr) + $offset), v)
                end)
        end
    end

    return GClasses[ty]
end

struct S2
    i::Int
end

struct S
    i::Int
    g::S2
end

r, v, b, s = GClass(S)

g = S(3, S2(4))
p = v(g)
p.i = 8
p.g.i = 223
p.g = S2(7)

println(p.g.i)