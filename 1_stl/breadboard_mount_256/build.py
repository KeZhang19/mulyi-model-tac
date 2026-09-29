"""Build new CAD from root_open_256, in original assembly coordinates.
Run from repository root: python build.py /tmp/new_mount_directory
"""
from pathlib import Path
import sys,json,hashlib
import cadquery as cq
import numpy as np

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
src=Path('1_stl/root_open_256')
n=np.array([-.019212307270460777,-.005271929917449214,.9998015273064405]);d=-6.866377852027004
mount_xy=[(-42.,216.),(33.,216.),(83.,-9.),(133.,-9.)]
post_xy=[(-43.,-11.5),(46.,221.)]
post_p=[np.array([x,y,(d-n[0]*x-n[1]*y)/n[2]]) for x,y in post_xy]
axis=post_p[1]-post_p[0];axis/=np.linalg.norm(axis)
base={name:cq.Shape.importBrep(str(src/f'{name}_cad.brep')) for name in ['upper_01','lower_03']}
upper=base['upper_01'];lower=base['lower_03']
cutters=[]
for x,y in mount_xy:
 shaft=cq.Solid.makeCylinder(3.4,60,cq.Vector(x,y,-41))
 pocket=cq.Solid.makeCylinder(6,40,cq.Vector(x,y,-16))
 cutters.extend([shaft,pocket])
lower=lower.cut(*cutters).clean()
posts=[];sockets=[]
for i,p in enumerate(post_p):
 # Short, flared root and lead-in tip; axis follows the actual parting plane.
 embedded=cq.Solid.makeCylinder(4.8,.8,cq.Vector(*(p-.8*n)),cq.Vector(*n))
 flare=cq.Solid.makeCone(4.8,4.,.8,cq.Vector(*p),cq.Vector(*n))
 stem=cq.Solid.makeCylinder(4.,4.2,cq.Vector(*(p+.8*n)),cq.Vector(*n))
 tip=cq.Solid.makeCone(4.,3.3,1.,cq.Vector(*(p+5*n)),cq.Vector(*n))
 post=embedded.fuse(flare,stem,tip).clean();posts.append(post)
 plane=cq.Plane(origin=cq.Vector(*(p-.1*n)),xDir=cq.Vector(*axis),normal=cq.Vector(*n))
 if i==0:
  socket=cq.Workplane(plane).circle(4.3).extrude(7.3).val()
  mouth=cq.Solid.makeCone(5.3,4.3,1.,cq.Vector(*(p-.1*n)),cq.Vector(*n))
 else:
  socket=cq.Workplane(plane).slot2D(10.6,8.6).extrude(7.3).val()
  mouth=(cq.Workplane(plane).slot2D(12.6,10.6)
         .workplane(offset=1.).slot2D(10.6,8.6).loft(combine=True).val())
 sockets.append(socket.fuse(mouth).clean())
lower=lower.fuse(*posts).clean();upper=upper.cut(*sockets).clean()
manifest={'source_directory':str(src),'unit':'mm','breadboard_pitch_mm':[25,25],
 'breadboard_edge_offset_mm':[25,25],'breadboard_actual_size_mm':None,
 'example_minimum_board_mm':[250,275],
 'lower_print_origin_on_example_board_mm':[7,15],
 'mount_holes':[], 'locators':[],
 'axis_normal_cad':n.tolist(),'slot_axis_cad':axis.tolist(),
 'parts':[], 'source_sha256':{},
 'standard_reference':{'url':'https://www.pgb-europe.com/en-gb/20652/socket-cap-screw-8-8-ft-iso4762-m-6x10','head_diameter_mm':10,'head_height_mm':6,'hex_key_mm':5},
 'design_choices':{'through_diameter_mm':6.8,'counterbore_diameter_mm':12,
 'counterbore_floor_above_bottom_mm':24,'assumed_washer_thickness_mm':0,
 'M6x30_projection_below_base_mm':6,'actual_board_thread_depth_verified':False,
 'pin_diameter_mm':8,'pin_height_mm':6,'socket_diameter_mm':8.6,'socket_depth_mm':7.2,
 'elongated_socket_length_mm':10.6,'pin_root_max_diameter_mm':9.6,
 'physical_fit_tested':False}}
for i,(x,y) in enumerate(mount_xy):
 z=(d-n[0]*x-n[1]*y)/n[2]
 manifest['mount_holes'].append({'id':f'M{i+1}','cad_xy_mm':[x,y],
  'lower_print_xy_mm':[x+60,y+19],'example_board_xy_mm':[x+67,y+34],
  'surface_height_above_base_mm':z+40,'counterbore_depth_at_axis_mm':z+16,
  'head_top_clearance_at_axis_mm':z+10})
for i,p in enumerate(post_p):
 manifest['locators'].append({'id':f'P{i+1}','cad_origin_mm':p.tolist(),
  'lower_print_xy_mm':(p[:2]+[60,19]).tolist(),'upper_socket':'round' if i==0 else 'elongated_along_post_baseline'})
for p in src.iterdir():
 if p.is_file():manifest['source_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
for name,s in [('upper_01',upper),('lower_03',lower)]:
 assert s.isValid() and len(s.Solids())==1
 s.exportBrep(str(out/f'{name}_cad.brep'));cq.exporters.export(s,str(out/f'{name}_cad.step'))
 box=s.BoundingBox();manifest['parts'].append({'name':name,'cad_valid':s.isValid(),'cad_solids':len(s.Solids()),
  'cad_volume_mm3':s.Volume(),'cad_faces':len(s.Faces()),'cad_bounds_mm':[[box.xmin,box.ymin,box.zmin],[box.xmax,box.ymax,box.zmax]],
  'print_rotation_x_deg':180 if name=='upper_01' else 0,
  'print_translation_mm':[60,230,34.8] if name=='upper_01' else [60,19,40]})
 print(name,manifest['parts'][-1],flush=True)
cq.exporters.export(cq.Compound.makeCompound([upper,lower]),str(out/'assembly_mount.step'))
# Hardware envelope is reference only and is excluded from the printable assembly.
heads=[]
for x,y in mount_xy:
 head=cq.Solid.makeCylinder(5,6,cq.Vector(x,y,-16))
 shank=cq.Solid.makeCylinder(3,30,cq.Vector(x,y,-46))
 heads.append(head.fuse(shank))
cq.exporters.export(cq.Compound.makeCompound(heads),str(out/'reference_m6x30_envelopes.step'))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('BUILD_DONE',out,flush=True)


from pathlib import Path
import json,sys
import cadquery as cq,numpy as np,trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
out=Path(sys.argv[1]);diagnostics=out/'diagnostics';diagnostics.mkdir(exist_ok=True)
for name in ['upper_01','lower_03']:
 shape=cq.Shape.importBrep(str(out/f'{name}_cad.brep'))
 BRepTools.Clean_s(shape.wrapped)
 mesher=BRepMesh_IncrementalMesh(shape.wrapped,.02,False,.08,False)
 vv=[];ff=[];ids=[]
 for fid,face in enumerate(shape.Faces()):
  loc=TopLoc_Location();poly=BRep_Tool.Triangulation_s(face.wrapped,loc);assert poly is not None
  offset=len(vv)
  for j in range(1,poly.NbNodes()+1):
   p=poly.Node(j).Transformed(loc.Transformation());vv.append([p.X(),p.Y(),p.Z()])
  for t in poly.Triangles():
   ix=[t.Value(k)+offset-1 for k in (1,2,3)];ff.append(ix[::-1] if face.wrapped.Orientation()==TopAbs_REVERSED else ix);ids.append(fid)
 mesh=trimesh.Trimesh(vertices=vv,faces=ff,process=True)
 counts=np.bincount(mesh.edges_unique_inverse)
 print(name,'triangles',len(mesh.faces),'vertices',len(mesh.vertices),'boundary',sum(counts==1),'nonmanifold',sum(counts>2),'watertight',mesh.is_watertight,flush=True)
 np.savez(diagnostics/f'{name}_raw_cad_mesh.npz',vertices=mesh.vertices,faces=mesh.faces)
 shape.exportBrep(str(diagnostics/f'{name}_meshed.brep'))


from pathlib import Path
import json,sys,collections
import cadquery as cq,numpy as np,trimesh,shapely
from OCP.BRep import BRep_Tool
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.gp import gp_Pnt,gp_Pnt2d
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON
out=Path(sys.argv[1]);manifest=json.loads((out/'manifest.json').read_text());report=[]
# This is the inherited source BSpline with the two known tessellator gaps.
ref_face=cq.Shape.importBrep('1_stl/trimmed_256/lower_03_cad.brep').Faces()[101]
surf=BRep_Tool.Surface_s(ref_face.wrapped);projector=ShapeAnalysis_Surface(surf)
for part in manifest['parts']:
 name=part['name'];raw=np.load(out/'diagnostics'/f'{name}_raw_cad_mesh.npz')
 mesh=trimesh.Trimesh(vertices=raw['vertices'],faces=raw['faces'],process=False)
 counts=np.bincount(mesh.edges_unique_inverse);boundary=mesh.edges_unique[counts==1]
 patches=[];neighbors=collections.defaultdict(list)
 for a,b in boundary:neighbors[int(a)].append(int(b));neighbors[int(b)].append(int(a))
 assert all(len(v)==2 for v in neighbors.values())
 pending=set(neighbors);loops=[]
 while pending:
  first=min(pending);loop=[first];prev=None;cur=first
  while True:
   nxt=next(x for x in neighbors[cur] if x!=prev)
   if nxt==first:break
   loop.append(nxt);prev,cur=cur,nxt
  pending.difference_update(loop);loops.append(loop)
 additions=[];edges={tuple(e) for e in mesh.edges.tolist()}
 for loop in loops:
  assert name=='lower_03'
  pts=mesh.vertices[loop];uv=[]
  for p in pts:
   q=projector.ValueOfUV(gp_Pnt(*p),1e-9);uv.append([q.X(),q.Y()])
  poly=shapely.Polygon(uv);assert poly.is_valid
  triangles=shapely.constrained_delaunay_triangles(poly)
  new=[]
  for tri in triangles.geoms:
   locids=[int(np.argmin(np.linalg.norm(np.array(uv)-q,axis=1))) for q in np.array(tri.exterior.coords)[:3]]
   new.append([loop[k] for k in locids])
  # All UV triangles share orientation. Reverse the entire patch if a boundary edge agrees.
  orientation=None
  for tri in new:
   for a,b in zip(tri,np.roll(tri,-1)):
    if (int(a),int(b)) in edges:orientation=True;break
    if (int(b),int(a)) in edges:orientation=False;break
   if orientation is not None:break
  assert orientation is not None
  if orientation:new=[t[::-1] for t in new]
  maxdist=0.;ns=0;outside=0;max_trimmed_distance=0.
  for tri in new:
   xyz=mesh.vertices[tri]
   for i in range(17):
    for j in range(17-i):
     p=xyz[0]*(1-(i+j)/16)+xyz[1]*i/16+xyz[2]*j/16
     q=projector.ValueOfUV(gp_Pnt(*p),1e-9);sp=surf.Value(q.X(),q.Y())
     maxdist=max(maxdist,float(np.linalg.norm(p-np.array([sp.X(),sp.Y(),sp.Z()]))));ns+=1
     classifier=BRepClass_FaceClassifier(ref_face.wrapped,gp_Pnt2d(q.X(),q.Y()),1e-7)
     if classifier.State() not in (TopAbs_IN,TopAbs_ON):outside+=1
     max_trimmed_distance=max(max_trimmed_distance,ref_face.distance(cq.Vertex.makeVertex(*p)))
  print('patch_samples',ns,'surface_distance',maxdist,'strict_uv_outside',outside,'trimmed_face_distance',max_trimmed_distance,flush=True)
  # The configured absolute tessellation tolerance is 0.02 mm. Retain the failed
  # exploratory 0.002 mm bounded-face check; do not call that stricter check passed.
  assert maxdist<.002 and max_trimmed_distance<.02,(maxdist,max_trimmed_distance)
  additions.extend(new);patches.append({'vertices':len(loop),'triangles':len(new),'sample_count':ns,'max_sampled_surface_deviation_mm':maxdist,'strict_uv_classifier_outside':outside,'max_sampled_trimmed_face_distance_mm':max_trimmed_distance,'exploratory_bounded_face_0p002mm_passed':max_trimmed_distance<.002,'acceptance_sampled_bounded_face_tolerance_mm':.02,'source_face':'trimmed_256/lower_03_cad.brep face 101','validation_note':'Keep both failed strict UV classification and the exploratory 0.002 mm gate. Accept local bounded-face samples against the explicitly configured 0.02 mm CAD mesh deflection. This is sampled local evidence, not a global mesh error certificate.'})
 if additions:mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=np.vstack([mesh.faces,additions]),process=False)
 counts=np.bincount(mesh.edges_unique_inverse)
 assert mesh.is_watertight and mesh.is_winding_consistent and not np.any(counts!=2)
 # Connected-component count without optional scipy/networkx.
 parent=np.arange(len(mesh.vertices))
 for a,b in mesh.edges_unique:
  while parent[a]!=a:a=parent[a]
  while parent[b]!=b:b=parent[b]
  parent[b]=a
 roots=set()
 for a in range(len(parent)):
  while parent[a]!=a:a=parent[a]
  roots.add(a)
 assert len(roots)==1
 if part['print_rotation_x_deg']==180:mesh.vertices*=np.array([1,-1,-1])
 mesh.vertices+=np.array(part['print_translation_mm'])
 path=out/f'{name}_print.stl';path.write_text(trimesh.exchange.stl.export_stl_ascii(mesh))
 reload=trimesh.load_mesh(path,process=True)
 assert reload.is_watertight and reload.is_winding_consistent
 assert np.all(reload.bounds[0]>-1e-5) and np.all(reload.extents[:2]<256)
 record={'name':name,'triangles':len(reload.faces),'vertices':len(reload.vertices),'watertight':reload.is_watertight,'winding_consistent':reload.is_winding_consistent,'connected_components':len(roots),'boundary_edges':0,'nonmanifold_edges':0,'volume_mm3':reload.volume,'cad_mesh_relative_volume_difference':abs(reload.volume-part['cad_volume_mm3'])/part['cad_volume_mm3'],'print_bounds_mm':reload.bounds.tolist(),'print_size_mm':reload.extents.tolist(),'recovered_source_patches':patches}
 part.update(record);report.append(record);print(json.dumps(record,indent=2),flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
(out/'diagnostics/mesh_validation.json').write_text(json.dumps(report,indent=2)+'\n')


# Compare every pre-existing face; only mating planes and the lower back may change.
face_report={}
for name in ['upper_01','lower_03']:
 old=cq.Shape.importBrep('1_stl/root_open_256/'+name+'_cad.brep')
 new=cq.Shape.importBrep(str(out/(name+'_cad.brep')))
 sig=[]
 for f in new.Faces():
  b=f.BoundingBox();sig.append([f.Area(),*f.Center().toTuple(),b.xmin,b.ymin,b.zmin,b.xmax,b.ymax,b.zmax])
 missing=[];matched=[]
 for i,f in enumerate(old.Faces()):
  b=f.BoundingBox();v=np.array([f.Area(),*f.Center().toTuple(),b.xmin,b.ymin,b.zmin,b.xmax,b.ymax,b.zmax])
  scores=np.max(np.abs(np.array(sig)-v),axis=1);best=int(np.argmin(scores))
  if scores[best]>1e-5:missing.append({'index':i,'type':f.geomType(),'area':f.Area(),'normal':f.normalAt().toTuple() if f.geomType()=='PLANE' else None,'signature':v.tolist()})
  else:matched.append(float(scores[best]))
 face_report[name]={'old_faces':len(old.Faces()),'new_faces':len(new.Faces()),'unchanged_faces':len(matched),'max_signature_error':max(matched),'changed_original_faces':missing}
(out/'diagnostics/face_preservation.json').write_text(json.dumps(face_report,indent=2)+'\n')

from pathlib import Path
import cadquery as cq,numpy as np,json,hashlib
out=Path(sys.argv[1]);m=json.loads((out/'manifest.json').read_text())
l=cq.Shape.importBrep(str(out/'lower_03_cad.brep'));u=cq.Shape.importBrep(str(out/'upper_01_cad.brep'))
r={'cad_parts':{},'fit':{},'holes':[],'root_opening':{}}
for part in m['parts']:
 name=part['name'];a=cq.importers.importStep(str(out/f'{name}_cad.step')).solids().vals();assert len(a)==1 and a[0].isValid()
 r['cad_parts'][name]={'step_valid':a[0].isValid(),'step_solids':len(a),'step_reimport_volume_mm3':a[0].Volume()}
a=cq.importers.importStep(str(out/'assembly_mount.step')).solids().vals();assert len(a)==2 and all(s.isValid() for s in a)
r['assembly_step_valid_solids']=len(a)
n=np.array(m['axis_normal_cad']);checks=[]
for delta in [0,.5,2,5,6.2,10]:
 volume=l.intersect(u.translate(tuple(delta*n))).Volume();assert abs(volume)<1e-5
 checks.append({'displacement_along_normal_mm':delta,'intersection_volume_mm3':volume})
r['fit']['opening_sweep_discrete']=checks
hardware=cq.importers.importStep(str(out/'reference_m6x30_envelopes.step')).solids().vals()
for s,h in zip(hardware,m['mount_holes']):
 for mold in [l,u]:assert abs(s.intersect(mold).Volume())<1e-5
 x,y=h['cad_xy_mm'];count=0
 for theta in np.linspace(0,2*np.pi,12,endpoint=False):
  for radius,z in [(3.35,-39.99),(3.35,-16.01),(5.95,-15.99)]:
   assert not l.isInside(cq.Vector(x+radius*np.cos(theta),y+radius*np.sin(theta),z),1e-7);count+=1
  assert l.isInside(cq.Vector(x+4.5*np.cos(theta),y+4.5*np.sin(theta),-16.01),1e-7);count+=1
 r['holes'].append({'id':h['id'],'void_and_bearing_land_probes':count,'hardware_intersection_mm3':0})
print('CAD, assembly, hardware checks passed',flush=True)
count=0
for x in np.linspace(-26.19,26.19,7):
 for z in np.linspace(-20.19,20.19,7):
  for y in [-25,-20,-19.0001,-18.999,-18,-12,-7.6]:
   p=cq.Vector(float(x),float(y),float(z));assert not u.isInside(p,1e-7) and not l.isInside(p,1e-7);count+=1
r['root_opening']={'width_mm':52.4,'height_mm':40.4,'empty_channel_probes':count}
xy=np.array([v['lower_print_xy_mm'] for v in m['mount_holes']]);assert np.allclose((xy-xy[0])/25,np.round((xy-xy[0])/25))
board=xy+np.array(m['lower_print_origin_on_example_board_mm']);size=np.array(m['example_minimum_board_mm']);assert np.all(board>=25) and np.all(board<=size-25)
assert np.allclose((board-25)/25,np.round((board-25)/25))
r['breadboard']={'relative_holes_share_25mm_grid':True,'example_edge_offset_mm':25,'example_minimum_mm':size.tolist(),'actual_size_verified':False}
r['face_preservation']=json.loads((out/'diagnostics/face_preservation.json').read_text());assert [q['index'] for q in r['face_preservation']['upper_01']['changed_original_faces']]==[3];assert [q['index'] for q in r['face_preservation']['lower_03']['changed_original_faces']]==[8,124]
r['mesh_validation']=json.loads((out/'diagnostics/mesh_validation.json').read_text())
unchanged=[]
for filename,expected in m['source_sha256'].items():
 actual=hashlib.sha256((Path(m['source_directory'])/filename).read_bytes()).hexdigest();assert actual==expected;unchanged.append(filename)
r['parent_files_hash_unchanged']=unchanged
r['validation_limits']='No physical printing, mating fit test, torque/load test, or actual breadboard size and blind thread depth measurement. Opening sweep uses discrete positions.'
(out/'validation.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
print('VALIDATION_DONE',flush=True)
