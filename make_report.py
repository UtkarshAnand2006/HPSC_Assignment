"""Create report.pdf and report.md only from measured CSV/validation data."""
from pathlib import Path
import csv,json,platform,hashlib,os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/laplace-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/laplace-cache")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
fontdir=Path(matplotlib.get_data_path())/"fonts"/"ttf"
for name,file in [("BodyFont","DejaVuSans.ttf"),("BodyFont-Bold","DejaVuSans-Bold.ttf"),("BodyFont-Italic","DejaVuSans-Oblique.ttf"),("BodyFont-BoldItalic","DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily("BodyFont",normal="BodyFont",bold="BodyFont-Bold",italic="BodyFont-Italic",boldItalic="BodyFont-BoldItalic")
from validate import read_mesh

root=Path('.');out=Path('results');ns=(100,1000,10000)
rows=[]
for n in ns:
    with open(out/f'runs_{n}.csv') as f:rows+=list(csv.DictReader(f))
assert len(rows)==60 and all(r['status']=='converged' and float(r['relative_residual'])<=1e-8 for r in rows)
val=json.loads((out/'validation.json').read_text())
idx={(int(r['N']),r['system'],r['storage'],r['method']):r for r in rows}
methods=('SOR','SD','MR','CG','BiCGSTAB')
for n in ns:
    for system in ('original','symmetrized'):
        for method in methods:
            a=idx[n,system,'Dense',method];b=idx[n,system,'CSR',method]
            assert a['omega']==b['omega']
with open(out/'all_runs.csv','w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,n in zip(axes,ns):
    xy,tri=read_mesh(n);ax.triplot(xy[:,0],xy[:,1],tri,lw=.24 if n==10000 else .45,color='#284a76')
    ax.set(title=f'N = {n:,}',xlabel='x',ylabel='y');ax.set_aspect('equal')
fig.tight_layout();fig.savefig(out/'meshes.png',dpi=210);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8.5,3.4))
xy,tri=read_mesh(10000)
for ax,suf,title in zip(axes,('', 's'),('Original system','Symmetrized, same b')):
    u=np.loadtxt(out/f'solution{suf}_10000.csv',delimiter=',',skiprows=1)[:,2]
    im=ax.tripcolor(xy[:,0],xy[:,1],tri,u,shading='gouraud',cmap='inferno',vmin=0,vmax=max(1,np.max(u)))
    ax.set(title=title,xlabel='x',ylabel='y');ax.set_aspect('equal');fig.colorbar(im,ax=ax)
fig.tight_layout();fig.savefig(out/'temperature.png',dpi=210);plt.close(fig)

styles=getSampleStyleSheet()
for style in styles.byName.values():
    style.fontName='BodyFont'
for name in ('Title','Heading1','Heading2','Heading3'):
    styles[name].fontName='BodyFont-Bold'
styles.add(ParagraphStyle(name='CellSmall',fontName='BodyFont',fontSize=8,leading=10))
styles.add(ParagraphStyle(name='CaptionSmall',fontName='BodyFont',fontSize=8.5,leading=11,textColor=colors.HexColor('#444444')))
styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=13
styles['Title'].fontSize=19;styles['Title'].leading=23
story=[];md=['# Laplace Equation on a Unit Square','Finite Element Assembly and Iterative Solver Comparison','', '11 October 2026','']

def p(s):story.append(Paragraph(s,styles['BodyText']));story.append(Spacer(1,7));md.extend([s.replace('<b>','').replace('</b>','').replace('<br/>','\n'),''])
def h(s):story.append(Paragraph(s,styles['Heading2']));story.append(Spacer(1,4));md.extend(['## '+s,''])
def table(headers,data,widths=None):
    content=[[Paragraph(str(x),styles['CellSmall']) for x in row] for row in [headers]+data]
    t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf5')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#526c87')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#aaaaaa'))]))
    story.append(t);story.append(Spacer(1,10))
    md.append('| '+' | '.join(map(str,headers))+' |');md.append('| '+' | '.join(['---']*len(headers))+' |')
    for row in data:md.append('| '+' | '.join(map(str,row))+' |')
    md.append('')
def page():story.append(PageBreak())
def image(file,w,height,caption):story.append(Image(str(out/file),width=w,height=height));story.append(Paragraph(caption,styles['CaptionSmall']));story.append(Spacer(1,7));md.extend([f'![{caption}](results/{file})',''])
def fmt(t):return f'{float(t):.6f}' if float(t)<10 else f'{float(t):.3f}'
def run_table(system):
    data=[]
    for n in ns:
        for method in methods:
            d=idx[n,system,'Dense',method];c=idx[n,system,'CSR',method]
            data.append([f'{n:,} / {method}',d['iterations'],c['iterations'],fmt(d['time_s']),fmt(c['time_s']),f'{float(d["time_s"])/float(c["time_s"]):.1f}x'])
    table(['N / method','Dense iter.','CSR iter.','Dense time (s)','CSR time (s)','Speedup'],data,[126,62,58,82,82,55])
def omega_table(system):
    table(['Nodes N','Empirical optimum omega','CSR SOR iterations'],[[f'{n:,}',f'{float(idx[n,system,"CSR","SOR"]["omega"]):.3f}',idx[n,system,'CSR','SOR']['iterations']] for n in ns],[105,245,115])

story.append(Paragraph('Laplace Equation on a Unit Square',styles['Title']))
story.append(Paragraph('Finite Element Assembly and Iterative Solver Comparison',styles['BodyText']))
story.append(Paragraph('11 October 2026 | Independently generated meshes and measured results',styles['CaptionSmall']))
story.append(Spacer(1,10))
h('1  Domain, boundary conditions and meshes')
p('The domain is 0 &lt; x &lt; 1, 0 &lt; y &lt; 1, with -div(grad T) = 0. Bottom and left edges have T = 0; top and right edges have T = 1. The conflicting values at (1,0) and (0,1) are assigned their mean, 0.5; (0,0) is 0 and (1,1) is 1. Corner values are needed for the discrete model; the continuum data have jumps at two corners.')
table(['Target nodes','Actual nodes N','P1 triangles','Boundary nodes','Interior DOFs'],[[f'{n:,}',f'{v["N"]:,}',f'{v["elements"]:,}',v['boundary'],n-v['boundary']] for n,v in zip(ns,val)],[90,95,95,90,95])
p('Gmsh 4.15.2 (algorithm 5) triangulates exactly N prescribed points: equally spaced boundary points and deterministic, stratified, jittered interior points. No extra nodes are inserted. The meshes are unstructured linear triangles, whereas the reference report uses quadrilaterals; its numerical values therefore should not be expected here.')
image('meshes.png',475,143,'Figure 1. The three Gmsh meshes; all triangle areas sum to one.')
p('<b>Assembly.</b> For a triangle of area Delta, K_ij = (beta_i beta_j + gamma_i gamma_j)/(4 Delta), with beta_i = y_j-y_k and gamma_i = x_k-x_j. Assemble all local matrices into K. Retain interior rows and replace boundary rows by unit rows to obtain A; b is zero inside and equals the prescribed temperature on boundary nodes. A is nonsymmetric after this row replacement.')
p('<b>Common convergence and initialization.</b> Every accepted run satisfies the true residual ||b-Ax||2/||b||2 &lt;= 10^-8. Maximum: 100,000 iterations. Original runs start with x_B = g and x_I = 0, so boundary residuals remain zero and SD/CG act on the SPD interior block. Symmetrized runs start at zero. Neither system uses a preconditioner. MR is one-step minimal residual, not GMRES or MINRES.')
page()
h('2  Original system: dense versus CSR')
run_table('original')
omega_table('original')
p('SOR omega is selected in CSR using a coarse search from 1.00 to 1.95 with step 0.05, followed by a local step-0.005 search within +/-0.05, clipped to [1,1.995]. The best factor minimizes iterations to the same tolerance; ties favor the smaller factor. This is an empirical grid optimum, not a proven continuous optimum. Search trials are retained in results/tuning_N.csv and are excluded from solver times.')
p('Both formats use the same method recurrences and full N-node equations. Dense matrix-vector products use one-thread OpenBLAS; CSR products and SOR sweeps use compiled C loops. The original CG and steepest-descent runs are justified by their invariant interior subspace, rather than by symmetry of the full A.')
p('SD: alpha = (r,r)/(r,Ar). MR: alpha = (Ar,r)/(Ar,Ar). Both update x by alpha*r. CG uses the standard three-term recurrence; BiCGSTAB uses the standard stabilized bi-conjugate gradient recurrence. Recursive residuals are checked against the true residual before accepting convergence.')
page()
h('3  Symmetrized system: As x = b')
p('As = (A + A^T)/2 is formed after the boundary rows have been imposed, with b retained exactly. Symmetrization is applied to the full matrix, including its boundary equations. These runs use the identical relative residual threshold, a zero initial guess and a separately tuned SOR factor.')
run_table('symmetrized')
omega_table('symmetrized')
p('Independent smallest-eigenvalue estimates are positive for all three As matrices (Section 5), so SD and CG have their usual SPD setting here. This observation is specific to these assembled systems: averaging an arbitrary matrix with its transpose does not guarantee positive definiteness.')
p('Small dense/CSR differences in Krylov iteration counts, especially BiCGSTAB, can arise from floating-point summation order. Every listed result is accepted using its true residual, and no failed run is hidden as a successful run.')
page()
h('4  Computing time, storage and implementation')
speed=[float(idx[n,s,'Dense',method]['time_s'])/float(idx[n,s,'CSR',method]['time_s']) for n in ns for s in ('original','symmetrized') for method in methods]
p(f'The measured dense/CSR speedup ranges from {min(speed):.1f}x to {max(speed):.1f}x. '+('CSR is faster in every listed run.' if min(speed)>1 else 'Some short runs are affected by timing overhead; see the individual ratios.')+' CSR traverses only stored nonzeros, while dense products traverse all N squared coefficients. These timings compare the stated implementations, not only abstract storage formats.')
data=[]
for v in val:
    n=v['N'];nz=v['nnz'];dense=8*n*n;csr=12*nz+4*(n+1)
    data.append([f'{n:,}',f'{nz:,}',f'{100*nz/n**2:.4f}',f'{dense/2**20:.3f}',f'{csr/2**20:.3f}',f'{dense/csr:.1f}x'])
table(['N','Nonzeros A','Density (%)','Dense (MiB)','CSR (MiB)','Storage ratio'],data,[65,95,85,80,80,60])
p('Storage above counts matrix payload only: dense float64 values versus CSR float64 values, int32 columns and int32 row pointers. It excludes work vectors, coordinates, allocator overhead, and the retained dense reference. These are representation sizes, not peak memory of the benchmarking program. As has '+', '.join(f'{v["nnz_s"]:,}' for v in val)+' nonzeros for the three mesh sizes, respectively.')
p('<b>Timing protocol.</b> C11, GCC 14.2.0, -O3 -march=native, IEEE double precision, one-thread OpenBLAS on an AMD EPYC 9V74 cloud CPU. Five repetitions for N=100 and three for N=1000 report the median; N=10000 has one full measured run per case. Timings include initial/final residual products and solver iterations; they exclude parsing, assembly, allocation, conversion, tuning and file output. Cloud scheduling and very short durations limit timing precision; reported speedups are machine-specific. The 10,000-node original-system times and residuals are retained at log precision (six decimal places / three significant digits); stopping uses the unrounded residual.')
h('5  What symmetrization changes')
table(['N','Min eigenvalue As','Max boundary error','Relative solution change'],[[f'{v["N"]:,}',f'{v["min_eigenvalue"]:.7g}',f'{v["sym_boundary_error"]:.6f}',f'{v["relative_solution_change"]:.6f}'] for v in val],[65,140,130,130])
p('The boundary row of As is no longer a unit equation: boundary/interior couplings are half of those in the original interior rows. Consequently the same b produces different boundary temperatures and a different solution. The maximum boundary error is max |xs_B-g|. Relative change is ||xs-x||2/||x||2, using independent sparse direct reference solves. Original boundary error is zero to numerical precision.')
table(['N','SOR A/As','SD A/As','MR A/As','CG A/As','BiCGSTAB A/As'],[[f'{n:,}']+[idx[n,'original','CSR',m]['iterations']+'/'+idx[n,'symmetrized','CSR',m]['iterations'] for m in methods] for n in ns],[60,80,85,85,70,85])
page()
h('6  Conclusion and independent validation')
table(['N','Direct residual A','Direct residual As','Interior analytic RMS'],[[f'{v["N"]:,}',f'{v["direct_residual"]:.3e}',f'{v["direct_residual_s"]:.3e}',f'{v["analytic_rms"]:.3e}'] for v in val],[65,135,135,130])
p('The independent Python check reconstructs each element matrix from mesh coordinates, verifies the total area equals one, checks row boundary conditions and As=(A+A^T)/2, and solves both systems with SciPy sparse direct LU. The maximum difference between independently assembled and C-assembled coefficients is '+f'{max(v["assembly_error"] for v in val):.3e}'+'. C CSR-CG solutions agree with the direct solutions to relative errors below '+f'{max(max(v["cg_direct_error"],v["cg_direct_error_s"]) for v in val):.3e}'+'. All 60 final solver/format/system/mesh runs converge; their recorded true residuals are retained in all_runs.csv.')
p('The analytic check uses the harmonic extension u_top(x,y)+u_right(x,y), with u_top = sum over odd k of 4/(k*pi) sin(k*pi*x) sinh(k*pi*y)/sinh(k*pi), and the right-edge term obtained by swapping x and y. Five hundred odd terms and a stable exponential evaluation are used. RMS is measured only at interior nodes of the original solution, because the symmetrized problem no longer imposes the original boundary values. Corner singularities and changing sample locations mean this is an accuracy check, not a controlled convergence-rate estimate.')
image('temperature.png',470,188,'Figure 2. Temperature fields on the 10,000-node mesh: original and symmetrized equations.')
best=[]
for n in ns:
    for s in ('original','symmetrized'):
        m=min(methods,key=lambda m:float(idx[n,s,'CSR',m]['time_s']));best.append(f'N={n:,}, {s}: {m}')
p('CSR sharply reduces storage and computing time as N grows. Krylov methods need substantially fewer iterations than unpreconditioned SD and MR here. The fastest measured CSR solver in each case is: '+'; '.join(best)+'. Empirical optimal SOR is competitive but requires a factor search. Symmetrization changes the physical boundary equations; a smaller residual for As is not evidence that the original heat problem has been solved.')
p('<b>Reproducibility.</b> Deliverables include solver.c, mesh_100.msh, mesh_1000.msh, mesh_10000.msh, this report, generate_meshes.py, validate.py, make_report.py, run.sh and README.md. Raw run CSVs, tuning histories, serialized CSR and Matrix Market matrices, b and nodal solutions are in results/. The reference PDF was used only to guide the section order and table layout; its values and authorship were not reused.')

def footer(canvas,doc):
    canvas.setFont('BodyFont',8);canvas.setFillColor(colors.HexColor('#555555'));canvas.drawString(42,25,'Laplace FEM | Independently measured results');canvas.drawRightString(A4[0]-42,25,str(doc.page))
SimpleDocTemplate('report.pdf',title='Laplace FEM Solver Comparison',author='',pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
Path('report.md').write_text('\n'.join(md).replace('&lt;','<').replace('&gt;','>')+'\n')
print('Created report.pdf and report.md from 60 measured runs.')
