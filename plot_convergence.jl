using PyPlot
using CSV, DataFrames, Statistics, Printf

function mean_convergence_rate(dxrange,vals)

    return mean(diff(log.(vals)) ./ diff(log.(dxrange)))

end

function plot_convergence(ax,dxrange,vals)
    ax.loglog(dxrange,vals,"-o",linewidth=2)
end

function plot_convergence(dxrange,vals;filename="",title="")

    fig,ax = PyPlot.subplots()
    for v in vals
        plot_convergence(ax,dxrange,v)
    end
    ax.grid()
    ax.set_xlabel("Element size")
    ax.set_ylabel("Relative error vs. element size")
    ax.set_title(title)
    # slope = mean_convergence_rate(dxrange,vals)
    # annotation = @sprintf "mean slope = %1.1f" slope
    # ax.annotate(annotation, (0.5,0.2), xycoords = "axes fraction")
    fig.tight_layout()
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end

end

filenames = ["stress-convergence-order-1.txt",
             "stress-convergence-order-2.txt",
             "stress-convergence-order-3.txt",
             "stress-convergence-order-4.txt"]
dfs = [CSV.read(f) for f in filenames]
vals = [df.s3 for df in dfs]

plot_convergence(dfs[1].dx,vals,title=L"Convergence of $\sigma_{12}$", filename = "s3-convergence.png")
