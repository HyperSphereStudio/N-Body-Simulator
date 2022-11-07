using CUDA
using StaticArrays
import Adapt

const FP = Float64

const GArray{T, N} = CuArray{T, N}
const GVec{T} = GArray{T, 1}
const GRef{T} = CuRef{T}

const CArray{T, N} = Array{T, N}
const CVec{T} = CArray{T, 1}
const CRef{T} = Ref{T}
const TStkArray{N, T} = MVector{N, T}  

TStkZeros(::Type{T}, N) where T = ntuple(x -> T(0), N)

macro CreateExecutionContext(useCPU)
    eval(quote 
        const UseCPU = $useCPU
        const TArray{T, N} = UseCPU ? CArray{T, N} : GArray{T, N}
        const TVec{T} = TArray{T, 1}
        const TRef{T} = UseCPU ? CRef{T} : GRef{T}
        const TRefArray{T, N} = GArray{GRef{T}, N}

        TRef(x::T) where T = TRef{T}(x)  

        if UseCPU
            eval(quote 
                    macro RegisterType(ty) end
                    function linearforeach(f::Function, container, shared_parameters...)
                        for c in 1:length(container)
                            f(c, shared_parameters...)
                        end
                    end
            end)
        else
            eval(quote 
                    macro RegisterType(ty)
                        return :(Adapt.@adapt_structure $ty)
                    end
                    function linearforeach(f::Function, container, shared_parameters...)
                        @cuda threads = length(container) f(shared_parameters...)
                    end
            end)
        end
    end)
end