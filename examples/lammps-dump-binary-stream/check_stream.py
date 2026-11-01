"""Small byte-layout and rejection checks for the narrow streaming example."""
import pathlib,struct,subprocess,sys,tempfile
binary=str(pathlib.Path(sys.argv[1]).resolve())
frame='''ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 20
0 30
ITEM: ATOMS id type x y z ix iy iz vx vy vz
1 2 1 2 3 0 0 0 .1 .2 .3
2 1 4 5 6 1 -1 0 .4 .5 .6
'''
with tempfile.TemporaryDirectory() as td:
 identity=pathlib.Path(td)/'identity';identity.write_text('2\n1 2\n2 1\n')
 def run(text,limit=0):return subprocess.run([binary,str(identity),str(limit)],input=text.encode(),capture_output=True)
 a=run(frame);assert a.returncode==0 and len(a.stdout)==56+2*9*8
 assert struct.unpack('<q6d',a.stdout[:56])==(0,0,10,0,20,0,30)
 assert struct.unpack('<18d',a.stdout[56:])==(1,2,3,0,0,0,.1,.2,.3,4,5,6,1,-1,0,.4,.5,.6)
 second=frame.replace('TIMESTEP\n0','TIMESTEP\n200');assert len(run(frame+second).stdout)==400
 invalid={
  'type':frame.replace('1 2 1 2 3','1 3 1 2 3'),
  'duplicate_id':frame.replace('2 1 4 5 6','1 1 4 5 6'),
  'nonfinite':frame.replace(' .1 .2 .3',' nan .2 .3'),
  'noninteger_image':frame.replace('3 0 0 0 .1','3 .5 0 0 .1'),
  'boundary':frame.replace('pp pp pp','pp pp ff'),
  'field':frame.replace('vx vy vz','fx fy fz'),
  'truncated':frame.rsplit('\n',2)[0]+'\n',
  'nonmonotonic':frame+frame,
  'changed_cell':frame+second.replace('0 20','0 21'),
  'extra_column':frame.replace(' .4 .5 .6',' .4 .5 .6 7')}
 for key,value in invalid.items():
  result=run(value);assert result.returncode!=0,key
 assert len(run(frame+second,1).stdout)==200
 print({'status':'PASS','valid_layout_checks':3,'negative_cases':list(invalid),'limit_check':True})
