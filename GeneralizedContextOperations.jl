using CUDA
using Cthulhu
using Expronicon
import Adapt

include("GeneralizedMemory.jl")

const FP = Float64

const GArray{T, N} = CuArray{T, N}
const GVec{T} = GArray{T, 1}
const GRef{T} = CuRef{T}

const CArray{T, N} = Array{T, N}
const CVec{T} = CArray{T, 1}
const CRef{T} = Ref{T}

TStkZeros(::Type{T}, N) where T = ntuple(x -> T(0), N)

macro CreateExecutionContext(useCPU)
   Core.eval(@__MODULE__, quote 
        const UseCPU = $useCPU
        const TArray{T, N} = UseCPU ? CArray{T, N} : GArray{T, N}
        const TVec{T} = TArray{T, 1}
        const TRef{T} = UseCPU ? CRef{T} : GRef{T}
        const TRefArray{T, N} = GArray{GRef{T}, N}

        TRef(x::T) where T = TRef{T}(x)  
        MakeRef(x::T) where T = UseCPU ? (ismutable(T) ? x : Ref(x)) : Ref(x)

        if UseCPU
            eval(quote 
                    macro RegisterType(ty) end
                    macro iterate(f, range, shared_parameters...)
                        fun = @eval(@__MODULE__, f)
                        Threads.@threads for c in range
                            fun(c, shared_parameters...)
                            return nothing
                        end
                    end
            end)
        else
            eval(quote 
                    macro RegisterType(ty)
                        return :(Adapt.@adapt_structure $ty)
                    end
                    macro iterate(f, range, shared_parameters...)
                        f = JLFunction(f)
                        idxVarName = f.args[1]
                        #Remove it 
                        deleteat!(f.args, 1)
                        insert!(f.body.args, 1,  :($idxVarName = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * step($range) + first($range)))
                        fun = codegen_ast(f)
                        
                        return esc(quote 
                            $fun
                            @device_code_warntype @cuda threads = length(range) $(f.name)($(shared_parameters...))
                        end)
                    end
            end)
        end
    end)
end