import torch, os, sys, cv2, json, argparse, random, glob, struct, math, time
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import numpy as np 
import os.path as osp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

import numpy as np

def sph_dir(theta, phi):
    """ Map spherical to Euclidean coordinates """
    st, ct = ek.sincos(theta)
    sp, cp = ek.sincos(phi)
    return Vector3f(cp*st, sp*st, ct)

def sph_convert(v):
    x2 = ek.pow(v.x, 2)
    y2 = ek.pow(v.y, 2)
    z2 = ek.pow(v.z, 2)

    r = ek.sqrt(x2+y2+z2)
    phi = ek.atan2(v.y, v.x)
    theta = ek.atan2(ek.sqrt(x2+y2), v.z)

    return r, theta, phi

def y_0_0(colors, theta, phi):
    K = ek.sqrt( 1.0/(4*ek.pi) )
    return K * colors

def y_1_n1(colors, theta, phi):
    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * ek.sin(phi) * -ek.sin(theta) * colors

def y_1_0(colors, theta, phi):
    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * ek.cos(theta) * colors

def y_1_p1(colors, theta, phi):
    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * ek.cos(phi) * -ek.sin(theta) * colors

def y_2_n2(colors, theta, phi):
    K = ek.sqrt( 15.0/(4*ek.pi) )
    return K * ek.sin(phi) * ek.cos(phi) * ek.pow( ek.sin(theta), 2 ) * colors

def y_2_n1(colors, theta, phi):
    K = ek.sqrt( 15.0/(4*ek.pi) )
    return K * ek.sin(phi) * ek.sin(theta) * -ek.cos(theta) * colors

def y_2_0(colors, theta, phi):
    K = ek.sqrt( 5.0/(16*ek.pi) )
    return K * ( 3*ek.pow( ek.cos(theta), 2 )-1 ) * colors

def y_2_p1(colors, theta, phi):
    K = ek.sqrt( 15.0/(4*ek.pi) )
    return K * ek.cos(phi) * ek.sin(theta) * -ek.cos(theta) * colors

def y_2_p2(colors, theta, phi):
    K = ek.sqrt( 15.0/(16*ek.pi) )
    return K * ( ek.pow( ek.cos(phi), 2 )-ek.pow( ek.sin(phi), 2 ) ) * ek.pow( ek.sin(theta), 2 ) * colors

class AuxIntegrator(SamplingIntegrator):
    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

        self.light = None
        self.light_radiance = None

    def sample(self, scene, sampler, ray, medium=None, active=True):
        result = Vector3f(0.0)
        si = scene.ray_intersect(ray, active)
        active = si.is_valid() & active

        # emitter = si.emitter(scene)
        # result = ek.select(active, Emitter.eval_vec(emitter, si, active), Vector3f(0.0))

        # z_axis = np.array([0, 0, 1])
        
        # vertex = si.p.numpy()
        # v_count = vertex.shape[0]
        # vertex = np.expand_dims(vertex, axis=1)
        # vertex = np.repeat(vertex, self.light.vertex_count(), axis=1)
        
        # light_vertices = self.light.vertex_positions_buffer().numpy().reshape(self.light.vertex_count(), 3)
        # light_vertices = np.expand_dims(light_vertices, axis=0)
        # light_vertices = np.repeat(light_vertices, v_count, axis=0)
        
        # sph_polygons = light_vertices - vertex
        # sph_polygons = sph_polygons / np.linalg.norm(sph_polygons, axis=2, keepdims=True)
        
        # z_axis = np.repeat( np.expand_dims(z_axis, axis=0), v_count, axis=0 )
        # result_np = np.zeros(v_count, dtype=np.double)
        
        # for idx in range( self.light.vertex_count() ):
        #     idx1 = (idx+1) % self.light.vertex_count()
        #     idx2 = (idx) % self.light.vertex_count()
            
        #     dp = np.sum( sph_polygons[:, idx1, :] * sph_polygons[:, idx2, :], axis=1 )
        #     acos = np.arccos(dp)
            
        #     cp = np.cross( sph_polygons[:, idx1, :], sph_polygons[:, idx2, :] )
        #     cp = cp / np.linalg.norm(cp, axis=1, keepdims=True)
            
        #     dp = np.sum( cp * z_axis, axis=1 )
            
        #     result_np += acos * dp
                        
        # result_np *= 0.5 * 1.0/math.pi
        # result_np = np.repeat( result_np.reshape((v_count, 1)), 3, axis=1 )
        
        # fin = self.light_radiance * Vector3f(result_np)
        # fin[fin < 0] = 0
        # result += ek.select(active, fin, Vector3f(0.0))

        ctx = BSDFContext()
        bsdf = si.bsdf(ray)

        # bs, bsdf_val = BSDF.sample_vec(bsdf, ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
        # pdf = bs.pdf
        # wo = si.to_world(bs.wo)

        ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active), True, active)
        pdf = ds.pdf
        active_e = active & ek.neq(ds.pdf, 0.0)
        wo = ds.d
        bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo, active_e)
        emitter_val[emitter_val > 1.0] = 1.0
        bsdf_val *= emitter_val
        
        # _, wi_theta, wi_phi = sph_convert(si.to_world(si.wi))
        _, wo_theta, wo_phi = sph_convert(wo)

        y_0_0_ = y_0_0(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        
        y_1_n1_ = y_1_n1(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_1_0_ = y_1_0(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_1_p1_ = y_1_p1(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))

        y_2_n2_ = y_2_n2(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_2_n1_ = y_2_n1(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_2_0_ = y_2_0(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_2_p1_ = y_2_p1(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))
        y_2_p2_ = y_2_p2(bsdf_val, wo_theta, wo_phi) / ek.select(pdf > 0, pdf, Float(0.01))

        return result, si.is_valid(), [ Float(y_0_0_[0]), Float(y_0_0_[1]), Float(y_0_0_[2]),\
                                        Float(y_1_n1_[0]), Float(y_1_n1_[1]), Float(y_1_n1_[2]),\
                                        Float(y_1_0_[0]), Float(y_1_0_[1]), Float(y_1_0_[2]),\
                                        Float(y_1_p1_[0]), Float(y_1_p1_[1]), Float(y_1_p1_[2]),\
                                        Float(y_2_n2_[0]), Float(y_2_n2_[1]), Float(y_2_n2_[2]),\
                                        Float(y_2_n1_[0]), Float(y_2_n1_[1]), Float(y_2_n1_[2]),\
                                        Float(y_2_0_[0]), Float(y_2_0_[1]), Float(y_2_0_[2]),\
                                        Float(y_2_p1_[0]), Float(y_2_p1_[1]), Float(y_2_p1_[2]),\
                                        Float(y_2_p2_[0]), Float(y_2_p2_[1]), Float(y_2_p2_[2]) ]

    def aov_names(self):
        names = []
        for i in range(0, 9):
            for c in ['r', 'g', 'b']:
                names.append('sh_%s_%d' % (c, i))
        
        return names

    def to_string(self):
        return "AuxIntegrator[]"

def compute_sh(obj_file, cam_pos, cam_lookat):
    light_radiance = 1.0

    register_integrator('auxintegrator', lambda props: AuxIntegrator(props))

    scene_template_file = './scene_template.xml'
    Thread.thread().file_resolver().append(os.path.dirname(scene_template_file))

    scene = load_file(scene_template_file, integrator='auxintegrator', fov="40", tx=cam_lookat[0], ty=cam_lookat[1], tz=cam_lookat[2], \
                        spp="100", width=200, height=200, obj_file=obj_file)

    # scene.integrator().light = load_string(LIGHT_TEMPLATE, lsx="1", lsy="1", lsz="1", lrx="0", lry="0", lrz="0", ltx="-1", lty="0", ltz="0", l=light_radiance)
    # scene.integrator().light_radiance = light_radiance

    scene.integrator().render(scene, scene.sensors()[0])
    film = scene.sensors()[0].film()
    film.set_destination_file('./render_output.exr')
    film.develop()

    sh_channels_list = []
    for i in range(0, 9):
        for c in ['r', 'g', 'b']:
            sh_channels_list.append('sh_%s_%d' % (c, i))

    f_sh = np.zeros((200, 200, 27), dtype=np.float)
    exrfile = pyexr.open('render_output.exr')

    for i, channel in enumerate(sh_channels_list):
        ch = exrfile.get(channel)
        f_sh[:, :, i:i+1] += ch
    
    return f_sh