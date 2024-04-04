module mpi_test

using MPI
using PartitionedArrays, IterativeSolvers
using PartitionedArrays: laplace_matrix, partition
using PartitionedSolvers: amg, preconditioner

function main()
    nodes_per_dir = (40,40,40)
    parts_per_dir = (2,2,1)
    nparts = prod(parts_per_dir)
    parts = LinearIndices((nparts,))
    @show parts
    A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
    @show A
    x_exact = pones(partition(axes(A,2)))
    b = A*x_exact

    t = PTimer(parts)
    x = similar(b,axes(A,2))
    x .= 0
    tic!(t)
    _, history = IterativeSolvers.cg!(x,A,b;log=true)
    
    @show history
    toc!(t, "without preconditioner")

    x .= 0
    tic!(t)
    Pl = preconditioner(amg(),x,A,b)
    _, history = IterativeSolvers.cg!(x,A,b;Pl,log=true)
    toc!(t, "with AMG preconditioner") 
    @show history

    display(t)
end

function time_test()
    with_mpi() do distribute
        np = 3
        ranks = distribute(LinearIndices((np,)))
        t = PTimer(ranks)
        tic!(t)
        map(ranks) do rank
            sleep(rank)
        end
        toc!(t,"Sleep")
        display(t)
    end
end

function test()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    print("Hello, I am rank $(rank) of size $(size).\n")
end

function test2()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    send_buf = nothing
    if rank == 0
        send_buf = collect(1:size) .* 100
        print("rank 0:\n $(send_buf)\n")
    end
    recv_buf = MPI.Scatter(send_buf, Int, comm; root=0)
    print("I got this on $(rank):\n  $(recv_buf)\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test2()
end

end