module coplot
using Plots, LaTeXStrings, LinearAlgebra

function main()
    N = [32, 48, 72, 108, 162, 243]
    e = [
        0.015671877408872836,
        0.010164610424028673,
        0.005981154927517239,
        0.003983586616171579,
        0.0026675543717195,
        0.0017678852944403827
    ]
    logN = log.(N)
    loge = log.(e)
    plt = plot(
        logN, loge, 
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ""
    )
    # linear regression
    A = [logN ones(length(N))]
    b = A\loge
    loge_reg = [b[1]*logN[i] + b[2] for i in 1:length(N)]
    plot!(plt, logN, loge_reg, linestyle = :dot, label = string("slope = ", round(b[1], sigdigits=3)))
    savefig(plt, "convergence.pdf")
end


end