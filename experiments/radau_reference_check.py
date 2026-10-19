import sys,numpy as np
sys.path.insert(0,'/home/claude/pdae/src')
from scipy.integrate import solve_ivp
from descdae import RCLadder, RCMesh, RCGraph
models={"2":RCLadder(N=6,Rc=1.0,Rs=1.0,Ra=1.0,Rg=2.0,a=1.0,b=2.0,V=4.0,omega=2.0,c0=1.0,cvar=0.3),"4":RCMesh(),"5":RCGraph()}
def ref(m,rtol,atol):
    T=m.T; rhs=lambda t,x:m.ftilde(x,t); jac=lambda t,x:m.schur(x,t); xi=np.zeros(m.n)
    for _ in range(600):
        s=solve_ivp(rhs,(0,T),xi,method="Radau",rtol=rtol,atol=atol,jac=jac,dense_output=True)
        xn=s.y[:,-1]; d=m.norm_P(xn-xi); xi=xn
        if d<1e-13: break
    ts=np.linspace(0,T,401); return s.sol(ts).T
for k,m in models.items():
    a=ref(m,1e-10,1e-12); b=ref(m,1e-13,1e-14)
    print(k, max(m.norm_P(u-v) for u,v in zip(a,b)))
