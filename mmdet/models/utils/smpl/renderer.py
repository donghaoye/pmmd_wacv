import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM']  = 'osmesa'
import pyrender

import torch
import numpy as np

import trimesh
import torch
from torch import sin, cos

from smplx import SMPL
import cv2

# colors = [
#     (0.5, 0.2, 0.2, 1.0),  # Defalut
#     # (.7, .5, .5, 1.),  # Pink
#     # (.7, .7, .6, 1.),  # Neutral
#     # (.5, .5, .7, 1.),  # Blue
#     # (.5, .55, .3, 1.),  # capsule
#     # (.3, .5, .55, 1.),  # Yellow
# ]


class Renderer(object):

    def __init__(self, focal_length=1000, height=512, width=512, faces=None, depth_only=False):
        self.focal_length = focal_length
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.faces = faces
        if self.faces is None:
            smpl = SMPL('data/smpl')
            self.faces = smpl.faces
        self.depth_only = depth_only

    def set_faces(self, faces):
        self.faces = faces

    def __call__(self, images, vertices, translation, colors=[(0.5, 0.2, 0.2, 1.0)], contact_labels=None):

        if colors == None:
            colors = [(0.5, 0.2, 0.2, 1.0)]

        # List to store rendered scenes
        output_images = []
        output_depthmaps = []

        # # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        # # For all iamges

        for i in range(len(images)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            verts = vertices[i].detach().cpu().numpy()
            mesh_trans = translation[i].cpu().numpy()
            verts = verts + mesh_trans[:, None, ] 
            num_people = verts.shape[0]

            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.5, 0.5, 0.5))

            # Create camera. Camera will always be at [0,0,0]
            # CHECK If I need to swap x and y
            camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
            camera_pose = np.eye(4)
            camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                      cx=camera_center[0], cy=camera_center[1])
            scene.add(camera, pose=camera_pose)
            # Create light source
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
            # for every person in the scene
            for n in range(num_people):
                mesh = trimesh.Trimesh(verts[n], self.faces)
                mesh.apply_transform(rot)
                trans = 0 * mesh_trans[n]
                trans[0] *= -1
                trans[2] *= -1

                # # render color for contact, need to debug...
                # if contact_labels is not None:
                #     hit_id = (contact_labels >= 0.5).nonzero()
                #     mesh.visual.vertex_colors = (191/255., 191/255., 191/255., 255/255.)
                #     # mesh.visual.vertex_colors[hit_id, :] = (255, 0, 0, 255)
                #     mesh.visual.vertex_colors[:3000, :] = (255/255., 0, 0, 255/255.)

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=colors[n % len(colors)]
                    )

                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                # # for sparse mesh
                # mesh = pyrender.Mesh.from_trimesh(
                #     mesh,
                #     material=material,
                #     wireframe=True,
                #     smooth=False
                #     )

                scene.add(mesh, 'mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)

            if not self.depth_only:
                # Alpha channel was not working previously need to check again
                # Until this is fixed use hack with depth image to get the opacity
                color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

                # color = color[::-1,::-1]
                # rend_depth = rend_depth[::-1,::-1]
                color = color.astype(np.float32) / 255.0
                valid_mask = (rend_depth > 0)[:, :, None]
                output_img = (color[:, :, :3] * valid_mask +
                            (1 - valid_mask) * img)
                output_img = np.transpose(output_img, (2, 0, 1))
                output_images.append(output_img)

            else:
                rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                output_depthmaps.append(rend_depth)
        
        if not self.depth_only:
            return output_images
        else:
            return output_depthmaps

    def delete(self):
        self.renderer.delete()

    
    def forward_360view(self, images, vertices, translation, colors=[(0.5, 0.2, 0.2, 1.0)], contact_labels=None):
        if colors == None:
            colors = [(0.5, 0.2, 0.2, 1.0)]

        # List to store rendered scenes
        output_images = []
        output_depthmaps = []

        # # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # # For all iamges

        for i in range(len(images)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            verts = vertices[i].detach().cpu().numpy()
            mesh_trans = translation[i].cpu().numpy()
            verts = verts + mesh_trans[:, None, ] 
            num_people = verts.shape[0]

            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.5, 0.5, 0.5))

            # Create camera. Camera will always be at [0,0,0]
            # CHECK If I need to swap x and y
            camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
            camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=camera_center[0], cy=camera_center[1])

            camera_pose = np.eye(4)
            scene.add(camera, pose=camera_pose)

            # num_angles = 360
            num_angles = 10
            for i in range(num_angles):
                each_angle = 360 / num_angles
                rot_360 = trimesh.transformations.rotation_matrix(np.radians(each_angle), [0, 0, 1])
                print(i, each_angle)

                # Create light source
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
                # for every person in the scene
                for n in range(num_people):
                    mesh = trimesh.Trimesh(verts[n], self.faces)
                    if i == 0:
                        mesh.apply_transform(rot)
                    mesh.apply_transform(rot_360)
                    trans = 0 * mesh_trans[n]
                    trans[0] *= -1
                    trans[2] *= -1

                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.2,
                        alphaMode='OPAQUE',
                        baseColorFactor=colors[n % len(colors)]
                        )

                    mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material)

                    scene.add(mesh, 'mesh')

                    # Use 3 directional lights
                    light_pose = np.eye(4)
                    light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                    scene.add(light, pose=light_pose)
                    light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                    scene.add(light, pose=light_pose)
                    light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                    scene.add(light, pose=light_pose)

                color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0
                valid_mask = (rend_depth > 0)[:, :, None]
                output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * img)
                output_img = np.transpose(output_img, (2, 0, 1))
                output_images.append(output_img)


        return output_images





import trimesh

def create_bounding_box_mesh(bbox_min, bbox_max):
    quad_vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]]
    ])

    faces = np.array([
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ])

    bbox_mesh = trimesh.Trimesh(vertices=quad_vertices, faces=faces, process=False)
    # bbox_mesh.remesh(face_reindexing=True)
    
    return bbox_mesh, quad_vertices, faces



def euler_to_rotation_matrix(euler_angles):
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    sr, sp, sy = sin(roll), sin(pitch), sin(yaw)
    cr, cp, cy = cos(roll), cos(pitch), cos(yaw)
    
    # Construct the rotation matrix
    rotation_matrix = torch.zeros(3, 3)
    rotation_matrix[0, 0] = cp * cy
    rotation_matrix[0, 1] = cp * sy
    rotation_matrix[0, 2] = -sp
    rotation_matrix[1, 0] = sr * sp * cy - cr * sy
    rotation_matrix[1, 1] = sr * sp * sy + cr * cy
    rotation_matrix[1, 2] = sr * cp
    rotation_matrix[2, 0] = cr * sp * cy + sr * sy
    rotation_matrix[2, 1] = cr * sp * sy - sr * cy
    rotation_matrix[2, 2] = cr * cp
    
    return rotation_matrix


class RendererBbox(object):

    def __init__(self, focal_length=1000, height=512, width=512, faces=None, depth_only=False):
        self.focal_length = focal_length
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.faces = faces
        if self.faces is None:
            smpl = SMPL('data/smpl')
            self.faces = smpl.faces
        self.depth_only = depth_only

    def set_faces(self, faces):
        self.faces = faces

    def __call__(self, images, vertices, translation, colors=[(.7, .7, .6, 1.)], contact_labels=None):

        if colors == None:
            colors = [(0.5, 0.2, 0.2, 1.0)]

        # List to store rendered scenes
        output_images = []
        output_depthmaps = []

        # # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        # rot_90 = trimesh.transformations.rotation_matrix(
        #     np.radians(90), [0, 1, 0])
        # # For all iamges

        for i in range(len(images)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            verts = vertices[i].detach().cpu().numpy()
            mesh_trans = translation[i].cpu().numpy()
            # verts = verts + mesh_trans[:, None, ] 

            verts_t = torch.from_numpy(verts)
            # euler_angles = torch.tensor([0, 90, 0]).float()
            euler_angles = torch.tensor([0, 0, 0]).float()
            right_side_rot = euler_to_rotation_matrix(euler_angles)
            verts_t = torch.einsum('bij,kj->bik', verts_t, right_side_rot)
            verts = verts_t.cpu().numpy()
            verts = verts + mesh_trans[:, None, ] 
            num_people = verts.shape[0]

            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.5, 0.5, 0.5))

            # Create camera. Camera will always be at [0,0,0]
            # CHECK If I need to swap x and y
            camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
            camera_pose = np.eye(4)
            camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                      cx=camera_center[0], cy=camera_center[1])
            scene.add(camera, pose=camera_pose)
            # Create light source
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)

            # for every person in the scene
            for n in range(num_people):
                mesh = trimesh.Trimesh(verts[n], self.faces)
                mesh.apply_transform(rot)
                
                trans = 0 * mesh_trans[n]
                trans[0] *= -1
                trans[2] *= -1

                # # render color for contact, need to debug...
                # if contact_labels is not None:
                #     hit_id = (contact_labels >= 0.5).nonzero()
                #     mesh.visual.vertex_colors = (191/255., 191/255., 191/255., 255/255.)
                #     # mesh.visual.vertex_colors[hit_id, :] = (255, 0, 0, 255)
                #     mesh.visual.vertex_colors[:3000, :] = (255/255., 0, 0, 255/255.)

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=colors[n % len(colors)]
                    )

                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                # # for sparse mesh
                # mesh = pyrender.Mesh.from_trimesh(
                #     mesh,
                #     material=material,
                #     wireframe=True,
                #     smooth=False
                #     )
                scene.add(mesh, 'mesh')

                # bbox_mesh.apply_transform(rot)
                # trans = 0 * mesh_trans[n]
                # trans[0] *= -1
                # trans[2] *= -1
                # body_material = pyrender.MetallicRoughnessMaterial(
                #     metallicFactor=0.2,
                #     alphaMode='OPAQUE',
                #     baseColorFactor=[0, 0, 1, 1.0],
                #     roughnessFactor=1
                #     )
                # bbox_mesh = pyrender.Mesh.from_trimesh(
                #     bbox_mesh,
                #     material=body_material,
                #     wireframe=True,
                #     smooth=False
                #     )
                # scene.add(bbox_mesh, 'bbox_mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)


            if not self.depth_only:
                # Alpha channel was not working previously need to check again
                # Until this is fixed use hack with depth image to get the opacity
                color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                # color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.ALL_WIREFRAME)

                # color = color[::-1,::-1]
                # rend_depth = rend_depth[::-1,::-1]
                color = color.astype(np.float32) / 255.0
                valid_mask = (rend_depth > 0)[:, :, None]
                output_img = (color[:, :, :3] * valid_mask +
                            (1 - valid_mask) * img)
                output_img = np.transpose(output_img, (2, 0, 1))
                output_images.append(output_img)

            else:
                rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                output_depthmaps.append(rend_depth)
        
        if not self.depth_only:
            return output_images
        else:
            return output_depthmaps

    def delete(self):
        self.renderer.delete()

