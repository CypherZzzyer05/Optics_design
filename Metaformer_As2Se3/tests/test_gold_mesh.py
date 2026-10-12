"""Control the finite stack and validate independent field normalization."""
import unittest
from dataclasses import replace
import numpy as np

from fdtd_pipeline.config import load_config, PROJECT_ROOT
from gold_mesh_research import (GoldMeshSimulator, geometry_source, incident_squared,
                                field_profile, EPS0, MU0, C)


class GeometryBackend:
    def __init__(self): self.objects={}; self.selected=None; self.all_selected=False
    def switchtolayout(self): pass
    def selectall(self): self.all_selected=True
    def select(self,name): self.selected=name; self.all_selected=False
    def delete(self):
        if self.all_selected:self.objects.clear()
        else:del self.objects[self.selected]
        self.all_selected=False
    def add(self,properties=None,**values):
        values={**(properties or {}),**{k.replace('_',' '):v for k,v in values.items()}}
        name=values.get('name','FDTD'); self.objects[name]=dict(values); self.selected=name
    addfdtd=addrect=addmesh=addplane=addpower=add
    def set(self,key,value):self.objects[self.selected][key]=value
    def setnamed(self,name,key,value):self.objects[name][key]=value
    def getnamed(self,name,key):return self.objects[name].get(key,0.)
    def getnamednumber(self,name):return int(name in self.objects)
    def setglobalmonitor(self,*args):pass
    def cwnorm(self):pass
    def getresult(self,*args):return np.array([-16,-8,0,8,16])*1e-9


class GoldMeshTests(unittest.TestCase):
    def simulator(self,gold,vacuum=False):
        obj=object.__new__(GoldMeshSimulator)
        obj.settings=load_config(PROJECT_ROOT/'configs/as2se3_au_fine8.json').simulation
        obj.fdtd=GeometryBackend(); obj.gold=gold; obj.vacuum=vacuum
        obj.electric_wavelengths=(839.,927.); obj.magnetic_wavelengths=(839.,927.)
        obj.project_path=None; obj.field_names=[]
        return obj

    def test_bare_control_changes_only_metal_presence(self):
        gold,bare=self.simulator(True),self.simulator(False)
        g=geometry_source(); gold.build(g); bare.build(g)
        expected=dict(gold.fdtd.objects); expected.pop('back_metal')
        self.assertEqual(set(bare.fdtd.objects),set(expected))
        for name,properties in expected.items():
            self.assertEqual(set(bare.fdtd.objects[name]),set(properties))
            for key,value in properties.items():
                if isinstance(value,np.ndarray):
                    np.testing.assert_array_equal(bare.fdtd.objects[name][key],value)
                else:self.assertEqual(bare.fdtd.objects[name][key],value)
        self.assertAlmostEqual(bare.fdtd.objects['substrate']['z min'],-350e-9)
        self.assertAlmostEqual(bare.fdtd.objects['T']['z'],-1100e-9)
        self.assertIn('metal_mesh',bare.fdtd.objects)
        self.assertIn('global_mesh_limit',bare.fdtd.objects)
        self.assertTrue(bare.grid_matches_Au_layout)

    def test_vacuum_reference_removes_every_physical_scatterer(self):
        vacuum=self.simulator(False,True); vacuum.build(geometry_source())
        self.assertNotIn('substrate',vacuum.fdtd.objects)
        self.assertNotIn('back_metal',vacuum.fdtd.objects)
        self.assertFalse(any(x.startswith('si_') for x in vacuum.fdtd.objects))
        self.assertIn('nf_top_E',vacuum.fdtd.objects)
        self.assertIn('nf_mid_H',vacuum.fdtd.objects)

    def test_native_structure_check_rejects_changed_substrate_or_gold(self):
        bare=self.simulator(False); g=geometry_source(); bare.build(g)
        bare.fdtd.objects['substrate']['z min']=-3000e-9
        with self.assertRaisesRegex(ValueError,'substrate/z min'):
            bare.native_structure_check(g)
        bare.fdtd.objects['substrate']['z min']=-350e-9
        bare.fdtd.objects['back_metal']={}
        with self.assertRaisesRegex(ValueError,'Au presence'):
            bare.native_structure_check(g)

    def test_electric_and_magnetic_incident_normalization(self):
        intensity=np.array([.5*EPS0*C,.5*MU0*C])
        self.assertAlmostEqual(incident_squared(intensity[:1],'E')[0],1.)
        self.assertAlmostEqual(incident_squared(intensity[1:],'H')[0],1.)
        for bad in ([0.],[-1.],[float('nan')]):
            with self.assertRaises(ValueError):incident_squared(bad)

    def test_native_vector_frequency_sort_and_squared_sum(self):
        raw=np.ones((2,3,1,2,3),dtype=complex)
        raw[:,:,:,0,:]*=2
        dataset={'x':np.array([0,8])*1e-9,'y':np.array([0,4,8])*1e-9,
                 'z':np.array([100])*1e-9,'lambda':np.array([927,839])*1e-9,'E':raw}
        intensity=np.full(2,.5*EPS0*C)
        xyz,w,enhancement,denominator=field_profile(dataset,intensity,[839,927])
        self.assertEqual(enhancement.dtype,np.float32)
        self.assertTrue(np.all(enhancement[...,0]==3))
        self.assertTrue(np.all(enhancement[...,1]==12))
        np.testing.assert_allclose(w,[839,927])
        np.testing.assert_allclose(xyz['z'],[100])

    def test_field_frequency_and_vector_dimensions_cannot_be_silently_reshaped(self):
        dataset={'x':[0.],'y':[0.],'z':[0.],'lambda':np.array([800])*1e-9,
                 'E':np.ones((1,1,1,1,3))}
        intensity=[.5*EPS0*C]
        with self.assertRaisesRegex(ValueError,'frequencies'):
            field_profile(dataset,intensity,[900])
        dataset['E']=np.ones((1,3))
        with self.assertRaisesRegex(ValueError,'dimensions'):
            field_profile(dataset,intensity,[800])


if __name__=='__main__':unittest.main()
