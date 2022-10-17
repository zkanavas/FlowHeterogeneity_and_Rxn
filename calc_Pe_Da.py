import numpy as np

def calc_flowrate(Dm,L,Pe,porosity,xarea):
    Uave = (Pe*Dm)/L
    darcy_velocity = Uave*porosity
    flow_rate = darcy_velocity*xarea
    return Uave,darcy_velocity,flow_rate

def calc_Da(r,Uave,rho,porosity,M):
    n = (rho*(1-porosity))/M
    Da = (np.pi*r)/(Uave*n)
    return n,Da

def calc_Pe(Dm,L,darcy_velocity,porosity):
    Uave = darcy_velocity/porosity
    Pe = (Uave*L)/Dm
    return Uave, Pe

# samples = ["ket0.1ph3.1","ket0.1ph3.6","estaillades"]
# Dm = 7.5e-10 #[m2s-1]
# S = [5019.11,4344.99,13471] #[1/m]
# L = [np.pi/s for s in S] #[m]
# phi = [0.114,0.131,0.144514]
# r = [8.1e-4,2.56e-4,8.1e-4] # [mol m-2 s-1]
# rho_calcite = 2.71e3 #[kg m-3]
# M_calcite = 0.1 #[kg mol-1]
# est = [3.67309e-05,3.67309e-08,3.67309e-06] # [m/s]

# ket31 = [0.000103199,7.46674e-11,7.46674e-08,7.46674e-06]

# ket36 = [0.000102786,7.67869e-11,6.77176e-08,7.67869e-06]

# for ind, samp in enumerate(samples):
#     if samp == "estaillades":
#         darcy_velocity = [3.67309e-05,3.67309e-08,3.67309e-06] #[m/s]
#     elif samp == "ket0.1ph3.6":
#         darcy_velocity = [0.000102786,7.67869e-11,6.77176e-08,7.67869e-06] #[m/s]
#     elif samp == "ket0.1ph3.1":
#         darcy_velocity = [0.000103199,7.46674e-11,7.46674e-08,7.46674e-06] #[m/s]
#     else: 
#         print("wrong sample name :", samp)
#         continue
#     for velocity in darcy_velocity:
#         Uave, Pe = calc_Pe(Dm,L[ind],velocity,phi[ind])
#         n, Da = calc_Da(r[ind],Uave,rho_calcite,phi[ind],M_calcite)
#         print(samp, " Pe: ",Pe, " Da: ", Da, " K: ", Pe*Da)
#     Uave,q,flowrate = calc_flowrate(Dm,L[ind],desired_Pe,phi[ind],cross_sectional_area)
#     n,Da = calc_Da(r[ind],Uave,rho_calcite,phi[ind],M_calcite)
#     print(samp, " Uave: ", Uave, " q: ", q, " Q: ",flowrate, " Da: ", Da, " n: ", n)


# desired_Pe = 500e-6

# for ind, samp in enumerate(samples):
#     Uave,q,flowrate = calc_flowrate(Dm,L[ind],desired_Pe,phi[ind],cross_sectional_area)
#     n,Da = calc_Da(r[ind],Uave,rho_calcite,phi[ind],M_calcite)
#     print(samp, " Uave: ", Uave, " q: ", q, " Q: ",flowrate, " Da: ", Da, " n: ", n)

desired_Pe = [0.0005]
Dm = 7.5e-10 #[m2s-1]
S = 5022.58
L = np.pi/S
phi = 5.78967/100
cross_sectional_area = 1.6473e-5 #m^2#(1000*3e-6)**2 # [m2]
print(cross_sectional_area)
r = 8.1e-4
rho_calcite = 2.71e3 #[kg m-3]
M_calcite = 0.1 #[kg mol-1]
samp = "ket0.1ph3.1"
for Pe in desired_Pe:
    Uave,q,flowrate = calc_flowrate(Dm,L,Pe,phi,cross_sectional_area)
    n,Da = calc_Da(r,Uave,rho_calcite,phi,M_calcite)
    print(samp," Pe: ",Pe, " Uave: ", Uave, " q: ", q, " Q: ",flowrate, " Da: ", Da, " n: ", n)