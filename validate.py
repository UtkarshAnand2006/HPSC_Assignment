"""Independent assembly, sparse-direct/eigenvalue and solution validation."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.sparse.linalg import spsolve, eigsh

def read_mesh(n):
    lines=Path(f'mesh_{n}.msh').read_text().splitlines()
    k=lines.index('$Nodes');count=int(lines[k+1]); rows=[line.split() for line in lines[k+2:k+2+count]]
    tags={int(row[0]):i for i,row in enumerate(rows)}
    xy=np.array([[float(row[1]),float(row[2])] for row in rows])
    k=lines.index('$Elements');count=int(lines[k+1]);tri=[]
    for line in lines[k+2:k+2+count]:
        row=list(map(int,line.split()))
        if row[1]==2:tri.append([tags[tag] for tag in row[3+row[2]:]])
    return xy,np.array(tri)

def analytic(xy):
    # Harmonic extension: u_top(x,y)+u_right(x,y); Fourier sine series.
    # Stable sinh(k*y)/sinh(k) avoids overflow. Corners excluded in RMS check.
    x,y=xy.T;u=np.zeros(len(x))
    for odd in range(1,1000,2):
        k=odd*np.pi
        def ratio(t):return (np.exp(-k*(1-t))-np.exp(-k*(1+t)))/(-np.expm1(-2*k))
        u+=4/k*(np.sin(k*x)*ratio(y)+np.sin(k*y)*ratio(x))
    return u

def validate(n):
    xy,tri=read_mesh(n);A=mmread(f'results/A_{n}.mtx').tocsr();S=mmread(f'results/As_{n}.mtx').tocsr();b=np.loadtxt(f'results/b_{n}.txt')
    rr=[];cc=[];vv=[];area=0
    for ids in tri:
        pts=xy[ids];d=np.linalg.det(np.column_stack([np.ones(3),pts]));area+=abs(d)/2
        bx=np.array([pts[1,1]-pts[2,1],pts[2,1]-pts[0,1],pts[0,1]-pts[1,1]])
        by=np.array([pts[2,0]-pts[1,0],pts[0,0]-pts[2,0],pts[1,0]-pts[0,0]])
        ke=(np.outer(bx,bx)+np.outer(by,by))/(2*abs(d))
        for i in range(3):
            for j in range(3):rr.append(ids[i]);cc.append(ids[j]);vv.append(ke[i,j])
    K=sparse.coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr()
    boundary=np.any((xy<1e-12)|(xy>1-1e-12),axis=1);interior=~boundary
    B=K.tolil()
    for i in np.where(boundary)[0]:B.rows[i]=[i];B.data[i]=[1.]
    diff=(B.tocsr()-A);assembly_error=np.max(np.abs(diff.data),initial=0)
    assert assembly_error<1e-11
    assert abs(area-1)<1e-12
    assert np.max(np.abs((S-(A+A.T)*.5).data),initial=0)<1e-14
    zero=np.any(xy<1e-12,axis=1);one=np.any(xy>1-1e-12,axis=1)
    g=np.where(zero&one,.5,one.astype(float));assert np.array_equal(b,g*boundary)
    x=spsolve(A,b);xs=spsolve(S,b)
    sol=np.loadtxt(f'results/solution_{n}.csv',delimiter=',',skiprows=1)[:,2]
    sols=np.loadtxt(f'results/solutions_{n}.csv',delimiter=',',skiprows=1)[:,2]
    out=dict(N=n,elements=len(tri),boundary=int(boundary.sum()),nnz=A.nnz,nnz_s=S.nnz,
             area=area,assembly_error=assembly_error,
             min_eigenvalue=float(eigsh(S,k=1,which='SA',tol=1e-9,return_eigenvectors=False)[0]),
             original_boundary_error=float(np.max(np.abs(x[boundary]-g[boundary]))),
             sym_boundary_error=float(np.max(np.abs(xs[boundary]-g[boundary]))),
             relative_solution_change=float(np.linalg.norm(xs-x)/np.linalg.norm(x)),
             direct_residual=float(np.linalg.norm(A@x-b)/np.linalg.norm(b)),
             direct_residual_s=float(np.linalg.norm(S@xs-b)/np.linalg.norm(b)),
             cg_direct_error=float(np.linalg.norm(sol-x)/np.linalg.norm(x)),
             cg_direct_error_s=float(np.linalg.norm(sols-xs)/np.linalg.norm(xs)),
             analytic_rms=float(np.sqrt(np.mean((x[interior]-analytic(xy[interior]))**2))),
             minimum_temperature=float(x.min()),maximum_temperature=float(x.max()))
    assert out['min_eigenvalue']>0
    assert out['cg_direct_error']<1e-5 and out['cg_direct_error_s']<1e-5
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':
    result=[validate(n) for n in (100,1000,10000)]
    Path('results/validation.json').write_text(json.dumps(result,indent=2)+'\n')
