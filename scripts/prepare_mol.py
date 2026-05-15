import os
import pandas as pd
import subprocess

folder = './new_optimized'

make_folder = \
f"""
cd {folder}
for entry in ./* 
do
    if [ "$entry" == "./annotations.tsv" ]; then
        continue
    fi
    file_name=$(basename $entry)
    folder=${{file_name%.*}}
    mkdir $folder
    mv $file_name $folder
done
cd ..
"""

subprocess.run(make_folder, shell=True)

inp = pd.read_csv(os.path.join(folder, 'annotations.tsv'), sep='\t', index_col=0)

for root, _, files in os.walk(folder):
    if root == folder:
        continue
    for file in files:
        file_path = os.path.join(root, file)

        with open(file_path,'r') as f:
            data = f.readlines()
        if not data[3][-6:] == 'V2000\n':
            data[3] = data[3][:-6] + ' V2000\n'

        stop = int(data[3][:3]) + 3

        if not data[4][-5:] == '0  0\n':
            for i, line in enumerate(data):
                if i < 4:
                    continue

                if i > stop:
                    break

                data[i] = line[:-5] + '0  0  0  0  0  0  0  0  0  0  0  0\n'

        with open(file_path,'w') as f:
            f.writelines(data)

       	param_path = os.path.join(root, 'param.inp')
        uhf_path = os.path.join(root, '.UHF')
        charge_path = os.path.join(root, '.CHRG')

        charge = int(inp.loc[file, 'CHARGE'])
        uhf = int(inp.loc[file, 'SPIN STATE'] * 2)
        temp = inp.loc[file, 'TEMP']

        param = \
f"""$cmd xtb --input default.inp --define --copy KANYUT.xyz
$date 2025/02/04 at 13:20:28.984     
$chrg {charge}
$spin {uhf}
$gfn
   method=2
   scc=true
   periodic=false
   dispscale=1.0000000000000000
$scc
   maxiterations=2000
   temp=100
   broydamp=0.40000000000000002
$opt
   optlevel=normal
   microcycle=20
   maxcycle=0
   maxdispl=1.0000000000000000
   hlow=0.10000000000000000E-1
   hessian=old
   s6=20.000000000000000
   kstretch=0.40000000000000002
   kbend   =0.13000000000000000
   ktorsion=0.74999999999999997E-2
   koutofp =0.0000000000000000
   kvdw    =0.0000000000000000
   kes     =0.0000000000000000
   rcut    =8.3666002653407556
   ts=0
   tsroot=0
   exact rf=false
   average conv=false
$thermo
   temp={temp}
   sthr=50.000000000000000
   imagthr=-20.000000000000000
   scale=1.0000000000000000
$md
   temp=298.14999999999998
   time=50.000000000000000
   dump=50.000000000000000
   velo=0
   nvt=1
   skip=500
   step=4.0000000000000000
   hmass=4
   shake=2
   sccacc=2.0000000000000000
   forcewrrestart=0
$siman
   dump=1000.0000000000000
   n=3
   ewin=20.000000000000000
   temp=1000.0000000000000
   enan=0
   check=0
$hess
   sccacc=0.29999999999999999
   step=0.50000000000000001E-2
   scale=1.0000000000000000
$modef
   n=31
   step=1.0000000000000000
   updat=0.20000000000000001
   local=0
   vthr=0.0000000000000000
   prj=0
   mode=7
$cube
   step=0.40000000000000002
   pthr=0.50000000000000003E-1
$symmetry
   desy=0.10000000000000001
   maxat=200
$embedding
   at=7
   es=true
$write
   esp=false
   mos=false
   gbw=false
   tm mos=false
   tm basis=false
   lmo=false
   density=true
   spin population=false
   spin density=false
   fod=false
   fod population=false
   wiberg=true
   wbo fragments=false
   dipole=true
   charges=true
   mulliken=true
   orbital energies=true
   inertia=true
   distances=true
   angles=false
   torsions=false
   final struct=true
   geosum=true
   stm=false
   modef=false
   gbsa=false
   vib_normal_modes=false
   hessian.out=false
$external
$path
   nrun=3
   npoint=50
   anopt=3
   kpush=0.50000000000000003E-1
   kpull=-0.40000000000000001E-1
   ppull=0.50000000000000003E-1
   alp=0.69999999999999996
$split
   fragment: 1,1-78
$wall
   potential=polynomial
   alpha=30
   beta=6.0000000000000000
   temp=300.00000000000000
   autoscale=1.0000000000000000
   axisshift=3.5000000000000000
$reactor
   max=50
   density=1.0000000000000000 # in kg/L
   kpush=0.50000000000000003E-1
   alp=0.69999999999999996
$end
"""
        with open(param_path,'w') as f:
            f.write(param)
        with open(uhf_path,'w') as f:
            f.write(str(uhf))
        with open(charge_path,'w') as f:
            f.write(str(charge))
