import vecset_diffusion

from pathlib import Path
import blender_plots as bplt
import bpy
import math
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from matplotlib import pyplot as plt

def save_blender(output_path: Path, name='test'):
    if 'Cube' in bpy.context.scene.objects:
        bplt.blender_utils.delete(bpy.context.scene.objects['Cube'])
    print('saving', str(output_path / f'{name}.blend'))
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path / f'{name}.blend'))

def read_homefile():
    bpy.ops.wm.read_homefile()
    if 'Cube' in bpy.context.scene.objects:
        bplt.blender_utils.delete(bpy.context.scene.objects['Cube'])

def render_image(output_path, resolution, samples=100, engine='cycles'):
    scene = bpy.context.scene
    version = bpy.app.version

    if engine == 'cycles':
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples

    elif engine == 'eevee':
        #  EEVEE Next only available in with bpy 4.2+
        if version >= (4, 2, 0):
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        else:
            scene.render.engine = 'BLENDER_EEVEE'

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def create_camera(location, rotation):
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"])

    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
    bpy.context.scene.camera = bpy.data.objects['Camera']

def setup_scene(
        clear=False,
        camera_location=None,
        camera_rotation=None,
        resolution=None,
        transparent=True,
        sun_energy=30.,
        sun_angle=np.pi / 2
    ):
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"])

    if camera_location is None:
        camera_location = np.array([0, -5.321560, 2.042498]) * 0.6
    if camera_rotation is None:
        camera_rotation = [math.radians(68.4), 0., 0.]

    if clear:
        read_homefile()
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes["Scene"].cycles.samples = 256

    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"])
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = sun_energy
    bpy.data.objects["Sun"].data.angle = sun_angle
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5

    bpy.context.scene.render.film_transparent = transparent
    create_camera(camera_location, camera_rotation)
    if resolution is not None:
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]

def create_animation(surface_points, latents_per_step, autoencoder, output_dir):
    print('creating animation...')
    read_homefile()
    setup_scene(
        clear=True,
        sun_angle=math.radians(5.),
        sun_energy=4.,
        transparent=True,
        camera_location=[1.658887, 3.840743, 2.236191],
        camera_rotation=[1.064819, -0.000003, 2.723454],
        resolution=(256, 256),
    )
    bpy.context.scene.objects['Sun'].rotation_euler = [-0.333459, -0.230782, 0.149516]
    bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256

    output_dir.mkdir(parents=False, exist_ok=False)

    s = bplt.Scatter(
        surface_points.cpu(),
        color=[0, 0, 1],
        marker_scale=0.01,
        name='volume points',
    )
    s.base_object.rotation_euler = [np.pi / 2, 0, 0]
    for i, latents_ in enumerate(tqdm(latents_per_step)):
        occupancies_tensor = vecset_diffusion.geometry.batch_query(
            autoencoder.decode,
            latents_[None].cuda(),
            128,
            batch_size=100000,
            progress_bar=False,
        )
        verts, faces = vecset_diffusion.geometry.get_mesh(occupancies_tensor)
        mesh = bplt.blender_utils.new_mesh(verts, faces, name='shape')
        mesh.rotation_euler = [np.pi / 2, 0, 0]
        render_image(
            output_path=str(output_dir / f'step_{i}.png'),
            resolution=(256, 256),
            samples=20,
            engine='eevee',
        )

    files = sorted([f for f in os.listdir(output_dir) if f.startswith('step_') and f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

    frames = []
    for f in files:
        im = Image.open(output_dir / f).convert('RGBA')
        bg = Image.new('RGBA', im.size, (255,255,255,255))
        bg.paste(im, mask=im)
        frames.append(bg.convert('RGB'))
    frames[0].save(
        output_dir / 'animation.gif',
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=1e3 / 20
    )

def render_mesh(output_path, mesh=None, pointcloud=None):
    setup_scene(
        clear=True,
        sun_angle=math.radians(5.),
        sun_energy=4.,
        transparent=True,
        camera_location=[1.138393, 2.670955, 1.282095],
        camera_rotation=[1.130275, -0.000004, 2.741782],
        resolution=(256, 256),
    )

    if mesh is not None:
        mesh_obj = bplt.blender_utils.new_mesh(mesh.vertices, mesh.faces)
        mesh_obj.rotation_euler = [np.pi / 2, 0, 0]

    if pointcloud is not None:
        s = bplt.Scatter(
            pointcloud,
            color=[0, 0, 1],
            marker_scale=0.01,
            name=f'pointcloud',
        )
        s.base_object.rotation_euler = [np.pi / 2, 0, 0]

    render_image(
        output_path=str(output_path),
        resolution=(256, 256),
        samples=20,
    )

def make_image_grid(output_path, images, image_names, eval_metrics=None, fig_title=None):
    num_rows = int(np.ceil(len(images) / 5))
    fig, axes = plt.subplots(num_rows, 5, figsize=(15.34,  3.26 * num_rows))

    for i, image in enumerate(images):
        ax = axes.ravel()[i]
        ax.imshow(image)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks([])
        ax.set_yticks([])

        if eval_metrics:
            cd = f'{eval_metrics[i]["chamfer_distance"] * 100:.2g}' \
            if eval_metrics[i]["chamfer_distance"] is not None else 'NaN'
            ca = f'{eval_metrics[i]["chamfer_unoriented_normal_angle"]:.2g}' \
                if eval_metrics[i]["chamfer_unoriented_normal_angle"] is not None else 'NaN'
            metrics_str = (
                f"\nCD: {cd} | "
                f"CA: {ca}"
                r"$^\circ$"
            )
        else:
            metrics_str = ""
        ax.set_title(f'{image_names[i]}' + metrics_str, fontsize=8)

    fig.tight_layout()
    if fig_title is not None:
        fig.suptitle(fig_title)
        fig.subplots_adjust(top=0.9)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
