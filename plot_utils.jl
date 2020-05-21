using PyPlot
using HDGElasticity
include("full_dirichlet_bc.jl")

function plot_contour(func,x0,widths;
        npoints=100,filename="",cmap="plasma")

    xrange = range(x0[1],stop=x0[1]+widths[1],length=npoints)
    yrange = range(x0[2],stop=x0[2]+widths[2],length=npoints)
    vals = [func([x,y]) for y in yrange, x in xrange]
    fig,ax = PyPlot.subplots()
    clabel = ax.contourf(xrange,yrange,vals,cmap=cmap)
    fig.colorbar(clabel)
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

x0 = [0.0,0.0]
widths = [1.0,1.0]
alpha = 0.01
plot_contour(x->test_displacement_field(x,alpha)[2],x0,widths,filename="u2-disp.png")
