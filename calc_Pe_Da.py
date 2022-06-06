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

# samples = ["ket0.1ph3.1","est0.1ph3.1","ket0.1ph3.6"]
# Dm = 7.5e-10 #[m2s-1]
# L = [4.24e-4,2.51e-4,4.58e-4] # [m]
# phi = [0.110,0.102,0.161]
# cross_sectional_area = 1.26e-5 # [m2]
# r = [8.1e-4,8.1e-4,2.56e-4] # [mol m-2 s-1]
# rho_calcite = 2.71e3 #[kg m-3]
# M_calcite = 0.1 #[kg mol-1]

# desired_Pe = 500e-6

# for ind, samp in enumerate(samples):
#     Uave,q,flowrate = calc_flowrate(Dm,L[ind],desired_Pe,phi[ind],cross_sectional_area)
#     n,Da = calc_Da(r[ind],Uave,rho_calcite,phi[ind],M_calcite)
#     print(samp, " Uave: ", Uave, " q: ", q, " Q: ",flowrate, " Da: ", Da, " n: ", n)

desired_Pe = [0.1]
Dm = 7.5e-10 #[m2s-1]
L = np.pi/13006.2
phi = 0.1181
cross_sectional_area = 4.60e-6 # [m2]
r = 8.1e-4
rho_calcite = 2.71e3 #[kg m-3]
M_calcite = 0.1 #[kg mol-1]
samp = "estaillades"
for Pe in desired_Pe:
    Uave,q,flowrate = calc_flowrate(Dm,L,Pe,phi,cross_sectional_area)
    n,Da = calc_Da(r,Uave,rho_calcite,phi,M_calcite)
    print(samp," Pe: ",Pe, " Uave: ", Uave, " q: ", q, " Q: ",flowrate, " Da: ", Da, " n: ", n)