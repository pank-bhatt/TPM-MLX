"""Mesh extraction, clean-up, simplification, and glTF / USDZ export."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def is_planar_hard_surface(mesh: trimesh.Trimesh, sample_size: int = 1500) -> bool:
    """Detects whether a mesh exhibits dominant planar or orthogonal facets (e.g. cubes, mechanical parts)."""
    try:
        if len(mesh.faces) < 50:
            return True
        normals = mesh.face_normals
        if len(normals) > sample_size:
            indices = np.random.choice(len(normals), size=sample_size, replace=False)
            normals = normals[indices]
        abs_normals = np.abs(normals)
        max_c = np.max(abs_normals, axis=1)
        # If > 55% of face normals strongly align with major axes (> 0.90), it's a planar hard-surface asset
        return float(np.mean(max_c > 0.90)) >= 0.55
    except Exception:
        return False


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 3, adaptive: bool = True) -> trimesh.Trimesh:
    """Applies adaptive non-shrinking Taubin smoothing.

    Hard-surface planar objects (cubes, mechanical shapes) bypass smoothing to preserve razor-sharp edges.
    Organic characters and creatures receive gentle smoothing to eliminate voxel steps.
    """
    if adaptive and is_planar_hard_surface(mesh):
        logger.info("Adaptive mesh smoothing: Planar hard-surface detected; bypassing smoothing to preserve sharp geometric edges.")
        return mesh

    try:
        from trimesh.smoothing import filter_taubin
        smoothed = filter_taubin(mesh, lamb=0.35, nu=-0.38, iterations=iterations)
        if len(smoothed.faces) > 0 and len(smoothed.vertices) > 0:
            return smoothed
    except Exception as e:
        logger.debug(f"Taubin smoothing fallback ({e}); using original mesh.")
    return mesh


def clean_mesh(
    mesh: trimesh.Trimesh,
    min_component_ratio: float = 0.01,
) -> trimesh.Trimesh:
    """Removes disconnected floating artifacts and keeps the primary geometry."""
    try:
        components = mesh.split(only_watertight=False)
        if len(components) <= 1:
            return smooth_mesh(mesh, adaptive=True)

        # Sort components by face count descending
        components = sorted(components, key=lambda c: len(c.faces), reverse=True)
        primary = components[0]
        total_faces = len(mesh.faces)

        # Retain components that have at least min_component_ratio of the total faces
        retained = [
            c for c in components
            if len(c.faces) >= total_faces * min_component_ratio
        ]
        if not retained:
            retained = [primary]

        combined = trimesh.util.concatenate(retained)
        return smooth_mesh(combined, adaptive=True)
    except Exception as e:
        logger.warning(f"clean_mesh failed ({e}); using original mesh.")
        return mesh


def simplify_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 40000,
) -> trimesh.Trimesh:
    """Simplifies mesh to target face count via decimation if face count exceeds target."""
    current_faces = len(mesh.faces)
    if current_faces <= target_faces or target_faces <= 0:
        return mesh

    try:
        # Attempt quadric decimation via trimesh
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if len(simplified.faces) > 0:
            return smooth_mesh(simplified, iterations=2, adaptive=True)
    except Exception as e:
        logger.debug(f"trimesh quadric decimation fallback ({e})")

    return mesh


def apply_studio_clay_material(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Applies a clean neutral studio clay PBR material for Stage 1 meshes."""
    try:
        material = trimesh.visual.material.PBRMaterial(
            name="StudioClay",
            baseColorFactor=[0.82, 0.82, 0.84, 1.0],  # Soft neutral studio clay
            metallicFactor=0.05,
            roughnessFactor=0.55,
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
    except Exception as e:
        logger.debug(f"Could not apply PBR clay material: {e}")
    return mesh


def bake_mesh_uv_texture(
    mesh: trimesh.Trimesh,
    reference_image: Optional[Any] = None,
    texture_size: int = 2048,
) -> trimesh.Trimesh:
    """Unwraps mesh into non-overlapping UVs via xatlas and bakes reference texture on Metal."""
    if reference_image is None:
        return apply_studio_clay_material(mesh)

    from PIL import Image
    from tpm_mlx.diffusion_3d.renderer.uvwrap_utils import mesh_uv_wrap
    from tpm_mlx.diffusion_3d.renderer.mesh_render_mlx import MeshRenderMLX

    if isinstance(reference_image, str):
        ref_pil = Image.open(reference_image)
    elif isinstance(reference_image, Image.Image):
        ref_pil = reference_image
    elif isinstance(reference_image, np.ndarray):
        ref_pil = Image.fromarray(reference_image)
    else:
        return apply_reference_texture_or_clay(mesh, reference_image)

    from PIL import ImageFilter
    from scipy.ndimage import distance_transform_edt

    ref_rgba = ref_pil.convert("RGBA")
    arr = np.array(ref_rgba)
    alpha = arr[..., 3]
    is_white_bg = (arr[..., 0] > 248) & (arr[..., 1] > 248) & (arr[..., 2] > 248)
    is_fg = (alpha > 30) & ~is_white_bg

    if np.any(is_fg) and not np.all(is_fg):
        # EDT foreground color dilation to eliminate white edge halo artifacts
        _, indices = distance_transform_edt(~is_fg, return_indices=True)
        dilated_arr = arr[indices[0], indices[1]]
        front_img = Image.fromarray(dilated_arr[..., :3])
    else:
        front_img = ref_rgba.convert("RGB")

    # Back view: horizontally mirrored and subtly shaded for natural depth
    back_arr = np.array(front_img.transpose(Image.FLIP_LEFT_RIGHT)).astype(np.float32) * 0.88
    back_img = Image.fromarray(np.clip(back_arr, 0, 255).astype(np.uint8))

    # Lateral side profiles: smooth color transition
    side_img = front_img.filter(ImageFilter.GaussianBlur(radius=4))

    # 1. UV parameterization
    unwrapped = mesh_uv_wrap(mesh)

    # 2. Metal UV-space rasterizer & texture baker
    render = MeshRenderMLX(default_resolution=1024, texture_size=texture_size)
    render.load_mesh(mesh=unwrapped)

    # Multi-angle 360 camera projection (Front, Right, Back, Left, Top-Front, Top-Back)
    baked_tex, mask = render.bake_texture(
        colors=[front_img, side_img, back_img, side_img, front_img, back_img],
        elevs=[0, 0, 0, 0, 35, 35],
        azims=[0, 90, 180, 270, 0, 180],
        weights=[1.0, 0.75, 0.7, 0.75, 0.4, 0.3],
    )

    # 3. Gutter inpainting and seam dilation via EDT
    baked_tex = render.uv_inpaint(baked_tex, mask)
    render.set_texture(baked_tex)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".obj") as tmp_obj:
        render.save_mesh(tmp_obj.name)

    textured_mesh = getattr(render, "_last_exported_mesh", None)
    if textured_mesh is not None:
        return textured_mesh

    return apply_reference_texture_or_clay(mesh, reference_image)


def apply_reference_texture_or_clay(
    mesh: trimesh.Trimesh,
    reference_image: Optional[Any] = None,
) -> trimesh.Trimesh:
    """Projects reference image colors onto the 3D mesh vertices, or applies studio clay."""
    if reference_image is None:
        return apply_studio_clay_material(mesh)

    try:
        from PIL import Image
        from scipy.ndimage import distance_transform_edt

        if isinstance(reference_image, str):
            ref_pil = Image.open(reference_image).convert("RGBA")
        elif isinstance(reference_image, Image.Image):
            ref_pil = reference_image.convert("RGBA")
        elif isinstance(reference_image, np.ndarray):
            ref_pil = Image.fromarray(reference_image).convert("RGBA")
        else:
            return apply_studio_clay_material(mesh)

        arr = np.array(ref_pil)
        H, W = arr.shape[:2]

        # 1. Foreground extraction & edge dilation (eliminates white edge fringing)
        alpha = arr[..., 3]
        is_white_bg = (arr[..., 0] > 245) & (arr[..., 1] > 245) & (arr[..., 2] > 245)
        is_fg = (alpha > 40) & ~is_white_bg

        if np.any(is_fg):
            _, indices = distance_transform_edt(~is_fg, return_indices=True)
            dilated_arr = arr[indices[0], indices[1]]
            coords = np.nonzero(is_fg)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
        else:
            dilated_arr = arr
            y_min, y_max = 0, H - 1
            x_min, x_max = 0, W - 1

        verts = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals) if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(verts) else None

        # 2. Normalized 3D coordinates (X=horizontal width, Y=vertical height, Z=depth)
        vx_min, vx_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        vy_min, vy_max = float(verts[:, 1].min()), float(verts[:, 1].max())

        span_x = max(1e-4, vx_max - vx_min)
        span_y = max(1e-4, vy_max - vy_min)

        u_norm = np.clip((verts[:, 0] - vx_min) / span_x, 0.0, 1.0)
        v_norm = np.clip(1.0 - (verts[:, 1] - vy_min) / span_y, 0.0, 1.0)

        # Front and back pixel coordinates
        px_front = np.clip(np.round(x_min + u_norm * (x_max - x_min)).astype(int), 0, W - 1)
        px_back = np.clip(np.round(x_max - u_norm * (x_max - x_min)).astype(int), 0, W - 1)
        py = np.clip(np.round(y_min + v_norm * (y_max - y_min)).astype(int), 0, H - 1)

        front_colors = dilated_arr[py, px_front].astype(np.float32)
        back_colors = dilated_arr[py, px_back].astype(np.float32) * 0.88

        # 3. Smooth bilateral blend across Z normal (depth axis) using smoothstep
        if normals is not None:
            nz = normals[:, 2]
            t = np.clip((nz + 0.15) / 0.30, 0.0, 1.0)
            # Smoothstep (3t^2 - 2t^3) for seamless boundary easing
            w = (t * t * (3.0 - 2.0 * t))[:, None]
            colors = (w * front_colors + (1.0 - w) * back_colors).astype(np.uint8)
        else:
            colors = front_colors.astype(np.uint8)

        colors[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
        return mesh
    except Exception as e:
        logger.warning(f"Vertex color projection note ({e}); falling back to studio clay.")
        return apply_studio_clay_material(mesh)


def export_mesh_files(
    mesh: trimesh.Trimesh,
    base_output_path: Union[str, Path],
    formats: Optional[list[str]] = None,
    reference_image: Optional[Any] = None,
) -> dict[str, str]:
    """Exports processed mesh to requested formats (.glb, .usdz, .obj).

    Args:
        mesh: Processed trimesh object.
        base_output_path: Path without file extension (or with .glb).
        formats: List of formats to export (default ["glb"]).
        reference_image: Optional reference PIL Image to project colors onto mesh.

    Returns:
        Dict mapping format string (e.g. 'glb', 'usdz') to file path.
    """
    formats = formats or ["glb"]
    p = Path(base_output_path)
    stem = p.stem
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Clean normals and orientation
    mesh.fix_normals()
    if getattr(getattr(mesh, "visual", None), "kind", "") != "texture":
        try:
            textured_mesh = bake_mesh_uv_texture(mesh, reference_image=reference_image, texture_size=2048)
            mesh.vertices = textured_mesh.vertices
            mesh.faces = textured_mesh.faces
            mesh.visual = textured_mesh.visual
        except Exception as e:
            logger.warning(f"UV texture bake note ({e}); falling back to vertex color projection.")
            mesh = apply_reference_texture_or_clay(mesh, reference_image=reference_image)

    results = {}

    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        target_file = parent / f"{stem}.{fmt_clean}"

        try:
            if fmt_clean == "glb":
                mesh.export(str(target_file), file_type="glb")
                results["glb"] = str(target_file)
            elif fmt_clean == "usdz":
                try:
                    mesh.export(str(target_file), file_type="usdz")
                    results["usdz"] = str(target_file)
                except Exception as ex:
                    logger.warning(f"USDZ export not natively supported by environment ({ex})")
            elif fmt_clean == "obj":
                mesh.export(str(target_file), file_type="obj")
                results["obj"] = str(target_file)
        except Exception as e:
            logger.error(f"Failed to export {fmt_clean}: {e}")

    return results
