import os, sys, math, random, argparse
import torch, cv2, trimesh

import numpy as np

import envmap_sh
import obj_sh

def detect_marker(img_rgb, img_depth):
    # Detect where to place the object in the image.
    # We may have to physically place a marker in the scene
    # Return x, y coordinates, specifing the pixel
    if 'img1' in img_path or 'img2' in img_path:
        return [352, 352]
    elif 'img3' in img_path:
        return [690, 188]

def compute_virtual_obj_sh(obj_file, cam_pos, cam_lookat):
    # Assume a simple material model for the object. For starters, make a uniform material (textures can be added later).
    # Compute SH coefs using MC integration. If uniform material, only 1 set of SH coefs. for the object.
    # This is potentially a precomputation, so doesnt matter if its computationally expensive.
    return obj_sh.compute_sh(obj_file, cam_pos, cam_lookat)

def compute_envmap_sh(img_rgb, img_depth):
    # Downscale image (4032x3024 --> 63x47)
    img_rgb = cv2.resize(img_rgb, (63, 47))

    return envmap_sh.compute_sh(img_rgb)

def detect_near_field(img_rgb, img_depth):
    # Use a simple YCbCr conversion, and threshold on the luma (Y) value.
    # Approximate with a polygon and return
    pass

def compute_near_field_sh(img_rgb, img_depth, light):
    # Use the following paper
    # "analytic spherical harmonic coefficients for polygonal area lights"
    pass

def compose(img_rgb, img_depth, virtual_obj_sh, env_map_sh, near_field_sh):
    # Depth aware compositing. Needs more thought, not very sure what to do.
    pass

def load_images(img_path, img_depth_path):
    img_rgb = cv2.cvtColor( cv2.imread(img_path), cv2.COLOR_BGR2RGB )
    img_depth = cv2.cvtColor( cv2.imread(img_depth_path), cv2.COLOR_BGR2RGB )

    # Normalize 0 to 1
    img_rgb = img_rgb / 255.0
    img_depth = img_depth / 255.0

    return img_rgb, img_depth

def compute_camera(img_path, img_rgb, img_depth):
    if 'img1' in img_path or 'img2' in img_path:
        cam_pos = [3, 3, 3]
        cam_lookat = [0, 0, 0]

        return cam_pos, cam_lookat
    elif 'img3' in img_path:
        cam_pos = [-0.5, 2, 4.3]
        cam_lookat = [0, 0, 0]

        return cam_pos, cam_lookat


if __name__ == '__main__':
    # Process one frame, can easily be extended to a video

    # -- Process command line args -- #
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image', type=str, help='')
    parser.add_argument('--obj_sh', type=str, default='NULL', help='')
    parser.add_argument('--obj_file', type=str, default='NULL', help='')

    args = parser.parse_args()

    # Assuming image_name.JPG is RGB and image_name_depth.JPG is depth 
    img_path = args.image
    img_depth_path = img_path.replace('.JPG', '_depth.JPG')

    # Load RGB and Depth image
    img_rgb, img_depth = load_images(img_path, img_depth_path)

    # Determine camera parameters
    cam_pos, cam_lookat = compute_camera(img_path, img_rgb, img_depth)

    # Detect loaction in the scene, in which to place the object
    obj_marker = detect_marker(img_rgb, img_depth)

    # SH coefs for virtual obj
    if args.obj_sh == 'NULL':
        virtual_obj_sh = compute_virtual_obj_sh(args.obj_file, cam_pos, cam_lookat)

        npy_file = args.obj_file.replace('.obj', '_sh.npy')
        np.save(npy_file, virtual_obj_sh)
    else:
        virtual_obj_sh = np.load(args.obj_sh)

    # Environment lighting SH
    env_map_sh = compute_envmap_sh(img_rgb, img_depth)
    env_map_sh = np.concatenate(env_map_sh, axis=0)
    env_map_sh = np.array([[env_map_sh]])
    env_map_sh = np.tile(env_map_sh, (200, 200, 1))

    obj_rendered = np.zeros((200, 200, 3), dtype=np.float)
    obj_rendered_ = env_map_sh * virtual_obj_sh
    for i in range(0, 9):
        obj_rendered[:, :, 0] += obj_rendered_[:, :, i*3]
        obj_rendered[:, :, 1] += obj_rendered_[:, :, i*3+1]
        obj_rendered[:, :, 2] += obj_rendered_[:, :, i*3+2]
    
    mask = img_rgb.copy()
    mask[:, :, :] = 0
    mask[obj_marker[0]-100:obj_marker[0]+100, obj_marker[1]-100:obj_marker[1]+100, :] = obj_rendered
    img_rgb[mask > 0.001] = 0

    obj_rendered = obj_rendered**(1.0/2.4)
    
    img_rgb[obj_marker[0]-100:obj_marker[0]+100, obj_marker[1]-100:obj_marker[1]+100, :] += obj_rendered
    final = np.clip(img_rgb, 0, 1) * 255.0
    final = cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite('render_re.png', final)

    # Near-field lighting
    # near_field_lights = detect_near_field(img_rgb, img_depth)
    # near_field_sh = []
    # for light in near_field_lights:
    #     sh = compute_near_field_sh(img_rgb, img_depth, light)
    #     near_field_sh.append(sh)
    
    # # Combine both types of lighting
    # final_augmented = compose(img_rgb, img_depth, virtual_obj_sh, env_map_sh, near_field_sh)

    # output_path = img_path.replace('.png', '') + '_augmented.png'
    # cv2.imsave(output_path, final_augmented)
