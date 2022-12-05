using Expronicon

include("GeneralizedMemory.jl")

TStkZeros(::Type{T}, N) where {T} = ntuple(x -> T(0), N)

macro CreateExecutionContext(useCPU)
    Core.eval(@__MODULE__,
        useCPU ?
        quote
            const TArray{T,N} = Array{T,N}
            const TRef{T} = Ref{T}

            MakeRef(x::T) where {T} = ismutable(T) ? x : Ref(x)

            macro RegisterType(ty) end
                macro iterate(f, range, shared_parameters...)
                    f = JLFunction(f)
                    return esc(quote
                        $(codegen_ast(f))
                        Threads.@threads for c in $range
                            $(f.name)(c, $(shared_parameters...))
                        end
                    end)
                end


        end :
        quote
            using CUDA
            using Cthulhu
            import Adapt

            CUDA.allowscalar(true)

            const TArray{T,N} = CuArray{T,N}
            const TRef{T} = CuRef{T}

            MakeRef(x::T) where {T} = Ref(x)

            macro RegisterType(ty)
                return :(Adapt.@adapt_structure $ty)
            end
            
            macro iterate(f, range, shared_parameters...)
                f = JLFunction(f)
                idxVarName = f.args[1]
                #Remove it 
                deleteat!(f.args, 1)
                insert!(f.body.args, 1, :($idxVarName = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * step($range) + first($range)))
                push!(f.body.args, :(return nothing))
                return esc(quote
                    $(codegen_ast(f))
                    @device_code_warntype @cuda threads = length(range) $(f.name)($(shared_parameters...))
                end)
            end
        end)

    Core.eval(@__MODULE__, quote
        const FP = Float64
        const TRefArray{T,N} = TArray{TRef{T}, N}
        const TVec{T} = TArray{T,1}

        TRef(x::T) where {T} = TRef{T}(x)
    end)
end