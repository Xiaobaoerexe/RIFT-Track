"""Faithfully author Unitree's URDF masses, full inertia and joint geometry into USD/MJCF."""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from pxr import Gf,Usd,UsdGeom,UsdPhysics

import argparse
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);args=p.parse_args()
assets=Path(args.assets).resolve();src=assets/'official/as2_description'
original=src/'urdf/as2.urdf';tree=ET.parse(original);robot=tree.getroot()
# Preserve the complete original file. Normalize just the root link's interface
# name to the existing robot/base sensor convention, with no inertial change.
for x in robot.iter():
    for k,v in list(x.attrib.items()):
        if v=='base_link':x.set(k,'base')
        if k=='filename':x.set(k,str(src/'meshes'/Path(v).name))
robot.set('name','unitree_as2')
working=assets/'as2_sim.urdf';tree.write(working,encoding='utf-8',xml_declaration=True)
stage=Usd.Stage.CreateNew(str(assets/'as2.usd'));UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
art=UsdGeom.Xform.Define(stage,'/Robot');stage.SetDefaultPrim(art.GetPrim());UsdPhysics.ArticulationRootAPI.Apply(art.GetPrim())
links={x.get('name'):x for x in robot.findall('link') if x.get('name')!='world'}
joints=[x for x in robot.findall('joint') if x.get('type')!='floating']
def vector(text,default='0 0 0'):return np.array([float(x) for x in (text or default).split()])
def origin(x):
    p=None if x is None else x.find('origin')
    return (np.zeros(3),np.eye(3)) if p is None else (vector(p.get('xyz')),Rotation.from_euler('xyz',vector(p.get('rpy'))).as_matrix())
def gfquat(R):
    x,y,z,w=Rotation.from_matrix(R).as_quat();return Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z)))
poses={'base':(np.zeros(3),np.eye(3))};pending=list(joints)
while pending:
    progress=False
    for j in list(pending):
        a,b=j.find('parent').get('link'),j.find('child').get('link')
        if a not in poses:continue
        t,R=origin(j);p,Q=poses[a];poses[b]=(p+Q@t,Q@R);pending.remove(j);progress=True
    if not progress:raise ValueError('Disconnected As2 graph')
def xform(geom,t,R):
    geom.AddTranslateOp().Set(Gf.Vec3d(*map(float,t)))
    x,y,z,w=Rotation.from_matrix(R).as_quat();geom.AddOrientOp().Set(Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z))))
inertials=[];joint_rows=[];collisions=0
for name,link in links.items():
    path='/Robot/'+name;body=UsdGeom.Xform.Define(stage,path);p,R=poses[name];xform(body,p,R)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());mass_api=UsdPhysics.MassAPI.Apply(body.GetPrim())
    node=link.find('inertial');mass=float(node.find('mass').get('value'));com,C=origin(node);a=node.find('inertia').attrib
    I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
    I=C@I@C.T;diag,axes=np.linalg.eigh(I)
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    assert mass>0 and np.all(diag>0),name
    mass_api.CreateMassAttr(mass);mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,com)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,diag)));mass_api.CreatePrincipalAxesAttr(gfquat(axes))
    inertials.append({'link':name,'mass_kg':mass,'com_m':com.tolist(),'inertia_kg_m2':I.tolist()})
    for kind in ('collision','visual'):
        UsdGeom.Xform.Define(stage,path+'/'+('collisions' if kind=='collision' else 'visuals'))
        for i,node in enumerate(link.findall(kind)):
            geom=node.find('geometry');g=next(iter(geom));gp=path+'/'+('collisions' if kind=='collision' else 'visuals')+f'/{kind}_{i}'
            if g.tag=='box':
                obj=UsdGeom.Cube.Define(stage,gp);obj.CreateSizeAttr(1.);obj.AddScaleOp().Set(Gf.Vec3f(*map(float,vector(g.get('size')))))
            elif g.tag=='sphere':obj=UsdGeom.Sphere.Define(stage,gp);obj.CreateRadiusAttr(float(g.get('radius')))
            elif g.tag=='cylinder':obj=UsdGeom.Cylinder.Define(stage,gp);obj.CreateRadiusAttr(float(g.get('radius')));obj.CreateHeightAttr(float(g.get('length')));obj.CreateAxisAttr('Z')
            elif g.tag=='mesh':
                mesh=trimesh.load(g.get('filename'),force='mesh',process=False);scale=vector(g.get('scale'),'1 1 1')
                obj=UsdGeom.Mesh.Define(stage,gp);obj.CreatePointsAttr((np.asarray(mesh.vertices)*scale).tolist());obj.CreateFaceVertexCountsAttr([3]*len(mesh.faces));obj.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist());obj.CreateSubdivisionSchemeAttr('none')
            else:raise ValueError(g.tag)
            t,Q=origin(node);xform(obj,t,Q)
            # Match USD transform evaluation: primitive scale acts in its local frame.
            if g.tag=='box':obj.SetXformOpOrder([obj.GetTranslateOp(),obj.GetOrientOp(),obj.GetScaleOp()])
            if kind=='collision':UsdPhysics.CollisionAPI.Apply(obj.GetPrim());collisions+=1
            else:obj.CreateDisplayColorAttr([Gf.Vec3f(.6,.65,.7)])
for j in joints:
    name=j.get('name');a,b=j.find('parent').get('link'),j.find('child').get('link');t,R=origin(j)
    path='/Robot/joints/'+name
    if j.get('type')=='fixed':obj=UsdPhysics.FixedJoint.Define(stage,path)
    else:
        obj=UsdPhysics.RevoluteJoint.Define(stage,path);axis=vector(j.find('axis').get('xyz'));index=int(abs(axis).argmax())
        assert axis[index]==1. and np.count_nonzero(axis)==1,name
        obj.CreateAxisAttr(('X','Y','Z')[index]);lim=j.find('limit').attrib
        obj.CreateLowerLimitAttr(math.degrees(float(lim['lower'])));obj.CreateUpperLimitAttr(math.degrees(float(lim['upper'])))
        joint_rows.append({'name':name,**{k:float(v) for k,v in lim.items()}})
    obj.CreateBody0Rel().SetTargets(['/Robot/'+a]);obj.CreateBody1Rel().SetTargets(['/Robot/'+b]);obj.CreateLocalPos0Attr(Gf.Vec3f(*map(float,t)));obj.CreateLocalRot0Attr(gfquat(R));obj.CreateLocalPos1Attr(Gf.Vec3f(0,0,0));obj.CreateLocalRot1Attr(Gf.Quatf(1.));obj.CreateCollisionEnabledAttr(False)
stage.GetRootLayer().Save()

report={"nominal_mass_kg":sum(x["mass_kg"] for x in inertials),"inertials":inertials,"joint_parameters":joint_rows,"collision_shapes":collisions,"usd_sha256":hashlib.sha256((assets/"as2.usd").read_bytes()).hexdigest(),"source_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),"physics_runtime_verified":False}
(assets/"asset_parameters.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ("inertials","joint_parameters")}))
