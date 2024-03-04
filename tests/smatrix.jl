using StaticArrays
const RealMatrix = SMatrix{2,2,Float64,4}

mutable struct Foo
    M::RealMatrix
end

function test1()
    M = zero(RealMatrix)
    R = randn(RealMatrix)
    for _ in 1:100000
        M += R
    end
end

function test2()
    M = zero(RealMatrix)
    f = Foo(M)
    R = randn(RealMatrix)
    for _ in 1:100000
        f.M += R
    end
end

@time test1()
@time test2()