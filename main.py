import os, sys, math, random
import torch, cv2, trimesh

def detect_marker(img_rgb, img_depth):
    pass


if __name__ == '__main__':
    # Process one frame, can easily be extended to a video

    # -- Process command line args -- #
    # Assuming image_name.png is RGB and image_name_depth.png is depth 
    img_path = sys.argv[1]
    img_depth_path = img_path.replace('.png', '') + '_depth.png'

    # Load obj model using trimesh.
    obj_path = sys.argv[2] 
    virtual_obj = trimesh.load(file_obj=obj_path, use_embree=True) # Embree because ray intersection tests might be required

    img_rgb = cv2.cvtColor( cv2.imread(img_path), cv2.COLOR_BGR2RGB )
    img_depth = cv2.cvtColor( cv2.imread(img_depth_path), cv2.COLOR_BGR2RGB )

    # Detect loaction in the scene, in which to place the object
    obj_marker = detect_marker(img_rgb, img_depth)

    # SH coefs for virtual obj
    virtual_obj_sh = compute_virtual_obj_sh(virtual_obj)

    # Environment lighting SH
    env_map_sh = compute_envmap_sh(img_rgb, img_depth)

    # Near-field lighting
    near_field_lights = detect_near_field(img_rgb, img_depth)
    near_field_sh = []
    for light in near_field_lights:
        sh = compute_near_field_sh(img_rgb, img_depth, light)
        near_field_sh.append(sh)
    
    # Combine both types of lighting
    final_augmented = compose(img_rgb, img_depth, virtual_obj_sh, env_map_sh, near_field_sh)

    output_path = img_path.replace('.png', '') + '_augmented.png'
    cv2.imsave(output_path, final_augmented)
