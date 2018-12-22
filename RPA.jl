using LinearAlgebra
using PyCall

@pyimport pylab as plt
@pyimport seaborn as sns

function step_func(x::Float64)
    if x > 0.0
        return 1.0
    else
        return 0.0
    end
end

function calc_ϵk_sq(kx::Float64, ky::Float64, t::Float64)
    return -2.0*t*(cos(kx) + cos(ky))
end

sq3 = sqrt(3.0)
function calc_ϵk_hy(kx::Float64, ky::Float64, t::Float64)
    return -sqrt(1.0 + 4.0*cos(0.5*ky)*cos(0.5*sq3*kx) + 4.0*cos(0.5*ky)^2.0)
end

function calc_ϵk(k::Float64)
    return 0.5*k^2
end

function calc_χ0_sq(qx, qy, ω, t, ϵF, Nk)
    χ0 = 0.0
    K = range(-pi, stop=pi, length=Nk)
    η = 0.1

    for i in 1:Nk
        kx = K[i]
        for j in 1:Nk
            ky = K[j]
            ϵk = calc_ϵk_sq(kx, ky, t)
            ϵkmq = calc_ϵk_sq(kx-qx, ky-qy, t)
            a = step_func(ϵF - ϵkmq) - step_func(ϵF - ϵk)
            b = ω + ϵkmq - ϵk + im*η
            χ0 += a/b 
        end
    end
    return -χ0/Nk/Nk
end

function calc_χ0_hy(qx::Float64, qy::Float64, ω::Float64, t::Float64, ϵF::Float64, Nk::Int64)
    a1 = [sqrt(3.0)/2.0; 1.0/2.0; 0.0]
    a2 = [sqrt(3.0)/2.0; -1.0/2.0; 0.0]
    a3 = [0.0; 0.0; 1.0]
    
    v = a1'*cross(a2, a3)
    g1 = 2.0*pi*cross(a2, a3)/v
    g2 = 2.0*pi*cross(a3, a1)/v

    χ0 = 0.0
    η = 0.01

    for i in 1:Nk
        k1 = (i - 1)/Nk*g1
        for j in 1:Nk
            k2 = (j - 1)/Nk*g2
            k = k1 + k2
            kx = k[1]
            ky = k[2]
            ϵk = calc_ϵk_hy(kx, ky, t)
            ϵkmq = calc_ϵk_hy(kx-qx, ky-qy, t)
            a = step_func(ϵF - ϵkmq) - step_func(ϵF - ϵk)
            b = ω + ϵkmq - ϵk + im*η
            χ0 += a/b 
        end
    end
    return -χ0/Nk/Nk
end

function calc_χ0_EG(q, ω, ϵF, Nk)
    χ0 = 0.0
    K = range(0.0, stop=50.0, length=Nk)
    η = 0.1

    # 極座標で積分
    for i in 1:Nk
        k = K[i]
        ϵk = calc_ϵk(k)
        ϵkmq = calc_ϵk(k - q)
        a = step_func(ϵF - ϵkmq) - step_func(ϵF - ϵk)
        b = ω + ϵkmq - ϵk + im*η
        χ0 += 4.0*pi*k^2.0*a/b 
    end
    return -χ0/Nk/Nk
end

function plot(ϵF, ω)
    Nq = 500
    Nk = 6
    Q = range(-pi, stop=pi, length=Nq)
    ϵk = zeros(Nq, Nq)
    χ0 = zeros(Complex, Nq, Nq)
    χzz = zeros(Complex, Nq, Nq)
    χN = zeros(Complex, Nq, Nq)
    Upt = 3.0
    t = 1.0
    U = Upt*t

    @time for j in 1:Nq
        @time for i in 1:Nq
            qx = Q[i]
            qy = Q[j]
            ϵk[i, j] = calc_ϵk_sq(qx, qy, t)
            χ = calc_χ0_sq(qx, qy, ω, t, ϵF, Nk)
            χ0[i, j] = χ
            χzz[i, j] = χ/(1 - U*χ)
            χN[i, j] = χ/(1 + U*χ)
        end
    end

    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.pcolormesh(Q, Q, real(χ0), cmap="magma")
    plt.colorbar()

    plt.subplot(222)
    plt.pcolormesh(Q, Q, real(χzz), cmap="magma")
    plt.colorbar()

    plt.subplot(223)
    plt.pcolormesh(Q, Q, real(χN), cmap="magma")
    plt.colorbar()

    plt.subplot(224)
    plt.contourf(Q, Q, ϵk, cmap="magma")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(string(ϵF)*"_"*string(ω)*".png")
    plt.show()

    s = ones(Nq)*(Q[2]-Q[1])
    for i in 1:Nq
        plt.plot(Q, real(χ0[i,:])+i*s,"black")
    end
    plt.show()
end

function plot_EG(ϵF, ω)
    Nq = 200
    Nk = 500
    Q = range(-pi, stop=pi, length=Nq)
    ϵk = zeros(Nq, Nq)
    χ0 = zeros(Complex, Nq, Nq)
    χzz = zeros(Complex, Nq, Nq)
    χN = zeros(Complex, Nq, Nq)

    for j in 1:Nq
        for i in 1:Nq
            qx = Q[i]
            qy = Q[j]
            q = sqrt(qx^2 + qy^2)
            ϵk[i, j] = calc_ϵk(q)
            χ0[i, j] = calc_χ0_EG(q, ω, ϵF, Nk)
        end
    end

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.pcolormesh(Q, Q, real(χ0), cmap="magma")
    plt.colorbar()

    plt.subplot(122)
    plt.contourf(Q, Q, ϵk, cmap="magma")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("FE"*string(ϵF)*"_"*string(ω)*".png")
    plt.show()

    #s = ones(Nk)*(Q[2] - Q[1])
    #for i in 1:100
    #    j = div(i*Nk, 100)
    #    plt.plot(Q, real(χ0[j,:])+j*s,"black")
    #end
    #plt.show()
end

function main1()
    NE = 10
    EF = range(0.0, stop=5.0, length=NE)
    ϵF= -1.0
    for i in 1:NE
        plot(ϵF,EF[i])
    end
end
#main1()

function main2()
    ϵF= -1.0
    ω = 0.0
    plot(ϵF, ω)
    plt.show()
end
main2()