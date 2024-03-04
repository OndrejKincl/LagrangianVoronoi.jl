module flotest

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using LinearAlgebra


const v = RealVector(-0.7844360502311856, -0.6886475957355536)
const a = RealVector(0.02149139863647087, -0.03722419436408397)
const b = -0.008775724091682273
const eps = 0.1

@fastmath function foo(x::RealVector)
    return dot(a,x) - b
end

function main()
    npos = 0
    nneg = 0
    nzero = 0
    for i in 1:100_000
        f = foo(v)
        s = sign(f)
        npos += Int(s == 1.0)
        nneg += Int(s == -1.0)
        nzero += Int(s == 0.0)
    end
    @show npos
    @show nneg
    @show nzero
end

main()

end