import genesis as gs
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- Inizializzazione del motore ---
gs.init(
    seed=None,
    precision='32',
    debug=True,
    eps=1e-12,
    logging_level='debug',
    backend=gs.cpu,
    theme='dark',
    logger_verbose_time='warning',
)

# --- Parametri globali ---
dt = 1e-2
debug = False
view = True

# --- Scena e opzioni di simulazione ---
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        substeps=500,                # più substeps per stabilità
        gravity=(0, 0, 0.0),         # gravità off per fase iniziale
    ),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_self_collision=True,
        enable_adjacent_collision=False,
        constraint_timeconst=0.1,    # più morbido, evita instabilità
        max_dynamic_constraints=10,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-1.5, -1.5, 0.0),
        upper_bound=(1.5, 1.5, 2.0),
        particle_size=0.005,         # meno particelle
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(640, 480),
        camera_pos=(0.5, 0, 0.3),
        camera_lookat=(0, 0, 0.1),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=debug,
        world_frame_size=0.5,
        show_link_frame=debug,
        show_cameras=False,
        plane_reflection=False,
        ambient_light=(0.1, 0.1, 0.1),
        shadow=False,
    ),
    show_viewer=view,
    renderer=gs.renderers.Rasterizer(),
    profiling_options=gs.options.ProfilingOptions(show_FPS=True),
)

# --- File del contenitore ---
DIR = "/home/edo/thesis"
container_mesh_path = DIR + '/becher/becher1.obj'

# --- Piano d'appoggio ---
plane = scene.add_entity(gs.morphs.Plane())

# --- Contenitore rigido con collisione ---
becher = scene.add_entity(
    gs.morphs.Mesh(
        file=container_mesh_path,
        fixed=True,
        pos=(0, 0, 0),
        euler=(90, 0, 180),
        scale=0.015,
        decimate=False,
        convexify=False,
        decompose_object_error_threshold=float("inf"),
        coacd_options=gs.options.CoacdOptions(),
        merge_submeshes_for_collision=True,
        collision=True,
    ),
    material=gs.materials.Rigid(needs_coup=True),
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture()
    ),
    visualize_contact=debug,
)

# --- Calcolo dimensioni del becher ---
container_mesh = trimesh.load(container_mesh_path)
container_bounds = container_mesh.bounds
container_size = (container_bounds[1] - container_bounds[0]) * 0.015

# --- Geometria del fluido ---
liquid_radius = min(container_size[0], container_size[1]) * 0.5 * 0.7
init_volume = 100e-6  # 100 mL
liquid_height = init_volume / (np.pi * liquid_radius**2) * 0.5
print(f"Radius: {liquid_radius*1e3:.2f} mm, Height: {liquid_height*1e3:.2f} mm")

# --- Materiale SPH con parametri scalati per dt=1e-2 ---
liquid = scene.add_entity(
    material=gs.materials.SPH.Liquid(
        rho=1000.0,
        stiffness=50.0,          # << ridotto per stabilità CFL
        exponent=7.0,
        mu=0.02,                 # viscosità aumentata per fase di settling
        gamma=0.0,               # tensione superficiale spenta inizialmente
        sampler='regular'
    ),
    morph=gs.morphs.Cylinder(
        pos=(0, 0, liquid_height / 2 + 0.002),
        radius=liquid_radius,
        height=liquid_height,
    ),
    surface=gs.surfaces.Default(
        color=(0.4, 0.8, 1.0),
        vis_mode='particle',
    ),
)

# --- Costruzione della scena ---
scene.build()

# --- Settling senza gravità ---
for _ in range(500):
    scene.step()

# --- Rampa graduale di gravità ---
n_ramp = 500
for i in range(n_ramp):
    g = -9.81 * (i + 1) / n_ramp
    scene._sim.options.gravity = (0, 0, g)
    scene.step()

# --- Parametri finali realistici ---
liquid._material.mu = 0.001
liquid._material.gamma = 0.005
liquid._material.stiffness = 200.0  # aumenta dopo la stabilizzazione

# --- Simulazione finale ---
for _ in range(2000):
    scene.step()
