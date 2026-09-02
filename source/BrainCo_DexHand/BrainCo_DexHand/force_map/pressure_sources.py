from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .pressure_taxel_map import (
    PenetrationFrame,
    PressureContactEventBatch,
    PressureTaxelMap,
    TaxelLayout,
    UrdfPressureContactAdapter,
)
from .urdf_pressure_layout import (
    UrdfPressurePadSpec,
    load_pressure_pad_specs_from_urdf,
    load_pressure_taxel_maps_from_urdf,
    load_pressure_touch_links_from_urdf,
    load_touch_links_from_urdf,
)


@dataclass(frozen=True)
class WarpSdfPenetrationSource:
    """Convert WarpSDF signed distances into canonical penetration frames."""

    contact_deadband_m: float = 0.0

    def frame_from_signed_distance(
        self,
        signed_distance_m: Any,
        *,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
    ) -> PenetrationFrame:
        signed_distance = _as_float_array(signed_distance_m)
        penetration = _clean_penetration(np.maximum(-signed_distance - float(self.contact_deadband_m), 0.0))
        return PenetrationFrame(
            penetration_m=penetration,
            signed_distance_m=signed_distance,
            contact_mask=penetration > 0.0,
            taxel_pose_w=taxel_pose_w,
            normal_w=normal_w,
        )


@dataclass(frozen=True)
class NormalRayPenetrationSource:
    """Convert normal-ray/deformation maps into canonical penetration frames.

    `frame_from_ray_distance` follows the common tactile convention where a
    smaller hit distance than the undeformed rest distance means indentation.
    `frame_from_deformation` accepts an already-computed deformation/depth map.
    """

    rest_distance_m: float | Any = 0.0
    contact_deadband_m: float = 0.0

    def frame_from_ray_distance(
        self,
        ray_distance_m: Any,
        *,
        valid_mask: Any | None = None,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
    ) -> PenetrationFrame:
        ray_distance = np.asarray(ray_distance_m, dtype=np.float32)
        finite = np.isfinite(ray_distance)
        ray_distance = np.where(finite, ray_distance, np.inf).astype(np.float32)
        rest_distance = np.asarray(self.rest_distance_m, dtype=np.float32)
        penetration = _clean_penetration(np.maximum(rest_distance - ray_distance - float(self.contact_deadband_m), 0.0))
        valid = finite
        if valid_mask is not None:
            valid = valid & np.asarray(valid_mask, dtype=bool)
        penetration = _clean_penetration(np.where(valid, penetration, 0.0))
        signed_distance = -penetration
        return PenetrationFrame(
            penetration_m=penetration,
            signed_distance_m=signed_distance.astype(np.float32),
            contact_mask=penetration > 0.0,
            taxel_pose_w=taxel_pose_w,
            normal_w=normal_w,
        )

    def frame_from_deformation(
        self,
        deformation_m: Any,
        *,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
    ) -> PenetrationFrame:
        deformation = _as_float_array(deformation_m)
        penetration = _clean_penetration(np.maximum(deformation - float(self.contact_deadband_m), 0.0))
        return PenetrationFrame(
            penetration_m=penetration,
            signed_distance_m=(-penetration).astype(np.float32),
            contact_mask=penetration > 0.0,
            taxel_pose_w=taxel_pose_w,
            normal_w=normal_w,
        )


@dataclass(frozen=True)
class SampledPenetrationStats:
    """Finite-area taxel sampling diagnostics in SI units."""

    penetration_m: np.ndarray
    support_fraction: np.ndarray
    active_count: np.ndarray
    mean_penetration_m: np.ndarray
    positive_mean_penetration_m: np.ndarray
    max_penetration_m: np.ndarray
    sample_penetrations_m: np.ndarray
    sample_offsets_l: np.ndarray
    sample_points_l_m: np.ndarray


@dataclass(frozen=True)
class GeometryNormalRayPenetrationSource:
    """Ray-cast simple object geometry along taxel normals.

    This source is intentionally independent from TacMap/Isaac. It returns the
    same canonical :class:`PenetrationFrame` as the TacMap-backed normal-ray path,
    but the ray hit distances come from explicit object geometry.
    """

    rest_distance_m: float | Any
    contact_deadband_m: float = 0.0
    max_distance_m: float | None = None
    ray_epsilon_m: float = 1.0e-9

    def frame_from_sphere(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        center_l: Any,
        radius_m: float | Any,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
    ) -> PenetrationFrame:
        distance, valid = self.ray_distance_to_sphere(
            ray_origins_l,
            ray_directions_l,
            center_l=center_l,
            radius_m=radius_m,
        )
        return self._frame_from_distance(distance, valid, taxel_pose_w=taxel_pose_w, normal_w=normal_w)

    def frame_from_box(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        center_l: Any,
        half_extents_l: Any,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
    ) -> PenetrationFrame:
        distance, valid = self.ray_distance_to_box(
            ray_origins_l,
            ray_directions_l,
            center_l=center_l,
            half_extents_l=half_extents_l,
        )
        return self._frame_from_distance(distance, valid, taxel_pose_w=taxel_pose_w, normal_w=normal_w)

    def frame_from_triangle_mesh(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
        chunk_size: int = 256,
    ) -> PenetrationFrame:
        distance, valid = self.ray_distance_to_triangle_mesh(
            ray_origins_l,
            ray_directions_l,
            vertices_l=vertices_l,
            triangles=triangles,
            chunk_size=chunk_size,
        )
        return self._frame_from_distance(distance, valid, taxel_pose_w=taxel_pose_w, normal_w=normal_w)

    def frame_from_closed_triangle_mesh_inside(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        taxel_pose_w: Any | None = None,
        normal_w: Any | None = None,
        chunk_size: int = 256,
        surface_sample_offsets_l: Any | None = None,
        sample_aggregation: str = "max",
        sample_min_support_fraction: float = 0.0,
    ) -> PenetrationFrame:
        if surface_sample_offsets_l is not None:
            stats = self.closed_triangle_mesh_inside_sampled_penetration_stats(
                ray_origins_l,
                ray_directions_l,
                vertices_l=vertices_l,
                triangles=triangles,
                surface_sample_offsets_l=surface_sample_offsets_l,
                sample_aggregation=sample_aggregation,
                sample_min_support_fraction=sample_min_support_fraction,
                chunk_size=chunk_size,
            )
            penetration = stats.penetration_m
            return PenetrationFrame(
                penetration_m=penetration,
                signed_distance_m=(-penetration).astype(np.float32),
                contact_mask=penetration > 0.0,
                taxel_pose_w=taxel_pose_w,
                normal_w=normal_w,
                sample_support_fraction=stats.support_fraction,
                sample_active_count=stats.active_count,
                sample_mean_penetration_m=stats.mean_penetration_m,
                sample_positive_mean_penetration_m=stats.positive_mean_penetration_m,
                sample_max_penetration_m=stats.max_penetration_m,
                sample_penetrations_m=stats.sample_penetrations_m,
                sample_offsets_l=stats.sample_offsets_l,
                sample_points_l_m=stats.sample_points_l_m,
            )

        penetration = self.closed_triangle_mesh_inside_penetration(
            ray_origins_l,
            ray_directions_l,
            vertices_l=vertices_l,
            triangles=triangles,
            chunk_size=chunk_size,
        )
        return PenetrationFrame(
            penetration_m=penetration,
            signed_distance_m=(-penetration).astype(np.float32),
            contact_mask=penetration > 0.0,
            taxel_pose_w=taxel_pose_w,
            normal_w=normal_w,
        )

    def closed_triangle_mesh_inside_penetration(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Return nearest-boundary penetration for ray origins inside a closed mesh."""

        forward_exit_distance, forward_inside = self.ray_exit_distance_from_closed_triangle_mesh(
            ray_origins_l,
            ray_directions_l,
            vertices_l=vertices_l,
            triangles=triangles,
            chunk_size=chunk_size,
        )
        backward_exit_distance, backward_inside = self.ray_exit_distance_from_closed_triangle_mesh(
            ray_origins_l,
            -_as_float_array(ray_directions_l),
            vertices_l=vertices_l,
            triangles=triangles,
            chunk_size=chunk_size,
        )
        # Local indentation should be the nearest boundary depth, not the full
        # thickness of the presser along one arbitrary ray direction.
        inside = forward_inside & backward_inside
        nearest_exit_distance = np.minimum(forward_exit_distance, backward_exit_distance)
        penetration = _clean_penetration(
            np.maximum(nearest_exit_distance - float(self.contact_deadband_m), 0.0)
        )
        return _clean_penetration(np.where(inside, penetration, 0.0))

    def closed_triangle_mesh_inside_sampled_penetration(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        surface_sample_offsets_l: Any,
        sample_aggregation: str = "max",
        sample_min_support_fraction: float = 0.0,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Sample finite taxel area around each ray origin and aggregate penetration."""

        return self.closed_triangle_mesh_inside_sampled_penetration_stats(
            ray_origins_l,
            ray_directions_l,
            vertices_l=vertices_l,
            triangles=triangles,
            surface_sample_offsets_l=surface_sample_offsets_l,
            sample_aggregation=sample_aggregation,
            sample_min_support_fraction=sample_min_support_fraction,
            chunk_size=chunk_size,
        ).penetration_m

    def closed_triangle_mesh_inside_sampled_penetration_stats(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        surface_sample_offsets_l: Any,
        sample_aggregation: str = "max",
        sample_min_support_fraction: float = 0.0,
        chunk_size: int = 256,
    ) -> SampledPenetrationStats:
        """Sample finite taxel area and return aggregate plus support diagnostics."""

        origins, directions, prefix = _ray_inputs(ray_origins_l, ray_directions_l)
        offsets = _surface_sample_offsets(surface_sample_offsets_l, prefix)
        sample_count = int(offsets.shape[1])
        if sample_count <= 0:
            zeros = np.zeros(prefix, dtype=np.float32)
            sample_zeros = np.zeros((*prefix, 0), dtype=np.float32)
            sample_vec_zeros = np.zeros((*prefix, 0, 3), dtype=np.float32)
            return SampledPenetrationStats(
                penetration_m=zeros,
                support_fraction=zeros,
                active_count=zeros,
                mean_penetration_m=zeros,
                positive_mean_penetration_m=zeros,
                max_penetration_m=zeros,
                sample_penetrations_m=sample_zeros,
                sample_offsets_l=sample_vec_zeros,
                sample_points_l_m=sample_vec_zeros,
            )

        sampled_origins = (origins[:, None, :] + offsets).reshape(-1, 3)
        sampled_directions = np.broadcast_to(directions[:, None, :], (origins.shape[0], sample_count, 3)).reshape(-1, 3)
        sampled = self.closed_triangle_mesh_inside_penetration(
            sampled_origins,
            sampled_directions,
            vertices_l=vertices_l,
            triangles=triangles,
            chunk_size=chunk_size,
        ).reshape(origins.shape[0], sample_count)

        mode = str(sample_aggregation).lower()
        active = sampled > 0.0
        active_count = np.count_nonzero(active, axis=1).astype(np.float32)
        support_fraction = active_count / float(sample_count)
        mean_penetration = np.mean(sampled, axis=1).astype(np.float32)
        summed = np.sum(np.where(active, sampled, 0.0), axis=1).astype(np.float32)
        positive_mean_penetration = summed / np.maximum(active_count, 1.0)
        max_penetration = np.max(sampled, axis=1).astype(np.float32)
        if mode == "max":
            penetration = max_penetration
        elif mode == "mean":
            penetration = mean_penetration
        elif mode == "positive_mean":
            penetration = positive_mean_penetration
        else:
            raise ValueError("sample_aggregation must be 'max', 'mean', or 'positive_mean'")
        min_support = max(0.0, min(1.0, float(sample_min_support_fraction)))
        if min_support > 0.0:
            penetration = np.where(support_fraction >= min_support, penetration, 0.0)
        return SampledPenetrationStats(
            penetration_m=_clean_penetration(penetration.reshape(prefix)),
            support_fraction=support_fraction.reshape(prefix).astype(np.float32),
            active_count=active_count.reshape(prefix).astype(np.float32),
            mean_penetration_m=_clean_penetration(mean_penetration.reshape(prefix)),
            positive_mean_penetration_m=_clean_penetration(positive_mean_penetration.reshape(prefix)),
            max_penetration_m=_clean_penetration(max_penetration.reshape(prefix)),
            sample_penetrations_m=_clean_penetration(sampled.reshape((*prefix, sample_count))),
            sample_offsets_l=offsets.reshape((*prefix, sample_count, 3)).astype(np.float32),
            sample_points_l_m=sampled_origins.reshape((*prefix, sample_count, 3)).astype(np.float32),
        )

    def ray_distance_to_sphere(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        center_l: Any,
        radius_m: float | Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        origins, directions, prefix = _ray_inputs(ray_origins_l, ray_directions_l)
        centers = _broadcast_vec3(center_l, prefix).reshape(-1, 3)
        radius = np.broadcast_to(np.asarray(radius_m, dtype=np.float32), prefix).reshape(-1)
        eps = max(float(self.ray_epsilon_m), 0.0)

        oc = origins - centers
        b = np.einsum("ij,ij->i", oc, directions)
        c = np.einsum("ij,ij->i", oc, oc) - radius * radius
        disc = b * b - c
        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t0 = -b - sqrt_disc
        t1 = -b + sqrt_disc
        hit = np.where(t0 > eps, t0, np.where(t1 > eps, t1, np.inf)).astype(np.float32)
        valid &= np.isfinite(hit)
        hit = self._apply_max_distance(hit, valid)
        return hit.reshape(prefix), np.isfinite(hit).reshape(prefix)

    def ray_distance_to_box(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        center_l: Any,
        half_extents_l: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        origins, directions, prefix = _ray_inputs(ray_origins_l, ray_directions_l)
        centers = _broadcast_vec3(center_l, prefix).reshape(-1, 3)
        half_extents = _broadcast_vec3(half_extents_l, prefix).reshape(-1, 3)
        bounds_min = centers - half_extents
        bounds_max = centers + half_extents
        eps = max(float(self.ray_epsilon_m), 1.0e-12)

        parallel = np.abs(directions) <= eps
        parallel_inside = np.all((origins >= bounds_min) & (origins <= bounds_max) | ~parallel, axis=-1)
        safe_dir = np.where(parallel, 1.0, directions)
        t1 = (bounds_min - origins) / safe_dir
        t2 = (bounds_max - origins) / safe_dir
        t_near_axis = np.where(parallel, -np.inf, np.minimum(t1, t2))
        t_far_axis = np.where(parallel, np.inf, np.maximum(t1, t2))
        t_near = np.max(t_near_axis, axis=-1)
        t_far = np.min(t_far_axis, axis=-1)
        valid = parallel_inside & (t_far >= np.maximum(t_near, eps))
        hit = np.where(valid, np.where(t_near > eps, t_near, t_far), np.inf).astype(np.float32)
        hit = self._apply_max_distance(hit, valid)
        return hit.reshape(prefix), np.isfinite(hit).reshape(prefix)

    def ray_distance_to_triangle_mesh(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        origins, directions, prefix = _ray_inputs(ray_origins_l, ray_directions_l)
        tri_vertices = _triangle_vertices(vertices_l, triangles)
        eps = max(float(self.ray_epsilon_m), 1.0e-12)
        chunk = max(1, int(chunk_size))

        v0 = tri_vertices[:, 0]
        edge1 = tri_vertices[:, 1] - v0
        edge2 = tri_vertices[:, 2] - v0
        best = np.full((origins.shape[0],), np.inf, dtype=np.float32)
        for start in range(0, origins.shape[0], chunk):
            stop = min(start + chunk, origins.shape[0])
            ray_o = origins[start:stop]
            ray_d = directions[start:stop]

            pvec = np.cross(ray_d[:, None, :], edge2[None, :, :])
            det = np.einsum("tj,rtj->rt", edge1, pvec, optimize=True)
            mask = np.abs(det) > eps
            inv_det = np.zeros_like(det, dtype=np.float32)
            inv_det[mask] = 1.0 / det[mask]

            tvec = ray_o[:, None, :] - v0[None, :, :]
            u = np.einsum("rtj,rtj->rt", tvec, pvec, optimize=True) * inv_det
            mask &= (u >= 0.0) & (u <= 1.0)

            qvec = np.cross(tvec, edge1[None, :, :])
            v = np.einsum("rj,rtj->rt", ray_d, qvec, optimize=True) * inv_det
            mask &= (v >= 0.0) & ((u + v) <= 1.0)

            t = np.einsum("tj,rtj->rt", edge2, qvec, optimize=True) * inv_det
            mask &= t > eps
            best[start:stop] = np.min(np.where(mask, t, np.inf), axis=1).astype(np.float32)

        best = self._apply_max_distance(best, np.isfinite(best))
        return best.reshape(prefix), np.isfinite(best).reshape(prefix)

    def ray_exit_distance_from_closed_triangle_mesh(
        self,
        ray_origins_l: Any,
        ray_directions_l: Any,
        *,
        vertices_l: Any,
        triangles: Any | None = None,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        origins, directions, prefix = _ray_inputs(ray_origins_l, ray_directions_l)
        tri_vertices = _triangle_vertices(vertices_l, triangles)
        eps = max(float(self.ray_epsilon_m), 1.0e-12)
        merge_tol = max(1.0e-7, eps * 10.0)
        chunk = max(1, int(chunk_size))

        v0 = tri_vertices[:, 0]
        edge1 = tri_vertices[:, 1] - v0
        edge2 = tri_vertices[:, 2] - v0
        exit_distance = np.full((origins.shape[0],), np.inf, dtype=np.float32)
        inside = np.zeros((origins.shape[0],), dtype=bool)
        for start in range(0, origins.shape[0], chunk):
            stop = min(start + chunk, origins.shape[0])
            ray_o = origins[start:stop]
            ray_d = directions[start:stop]

            pvec = np.cross(ray_d[:, None, :], edge2[None, :, :])
            det = np.einsum("tj,rtj->rt", edge1, pvec, optimize=True)
            mask = np.abs(det) > eps
            inv_det = np.zeros_like(det, dtype=np.float32)
            inv_det[mask] = 1.0 / det[mask]

            tvec = ray_o[:, None, :] - v0[None, :, :]
            u = np.einsum("rtj,rtj->rt", tvec, pvec, optimize=True) * inv_det
            mask &= (u >= 0.0) & (u <= 1.0)

            qvec = np.cross(tvec, edge1[None, :, :])
            v = np.einsum("rj,rtj->rt", ray_d, qvec, optimize=True) * inv_det
            mask &= (v >= 0.0) & ((u + v) <= 1.0)

            t = np.einsum("tj,rtj->rt", edge2, qvec, optimize=True) * inv_det
            mask &= t > eps
            if self.max_distance_m is not None:
                mask &= t <= float(self.max_distance_m)

            for local_i in range(stop - start):
                values = np.sort(t[local_i, mask[local_i]].astype(np.float32))
                if values.size == 0:
                    continue
                unique = [float(values[0])]
                for value in values[1:]:
                    if abs(float(value) - unique[-1]) > merge_tol:
                        unique.append(float(value))
                if len(unique) % 2 == 1:
                    inside[start + local_i] = True
                    exit_distance[start + local_i] = np.float32(unique[0])

        exit_distance = self._apply_max_distance(exit_distance, inside)
        return exit_distance.reshape(prefix), inside.reshape(prefix)

    def _frame_from_distance(
        self,
        ray_distance_m: np.ndarray,
        valid_mask: np.ndarray,
        *,
        taxel_pose_w: Any | None,
        normal_w: Any | None,
    ) -> PenetrationFrame:
        source = NormalRayPenetrationSource(
            rest_distance_m=self.rest_distance_m,
            contact_deadband_m=self.contact_deadband_m,
        )
        return source.frame_from_ray_distance(
            ray_distance_m,
            valid_mask=valid_mask,
            taxel_pose_w=taxel_pose_w,
            normal_w=normal_w,
        )

    def _apply_max_distance(self, distance: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if self.max_distance_m is None:
            return np.where(valid, distance, np.inf).astype(np.float32)
        max_distance = float(self.max_distance_m)
        finite = valid & (distance <= max_distance)
        return np.where(finite, distance, np.inf).astype(np.float32)


def triangle_mesh_topology_diagnostics(
    vertices_l: Any,
    triangles: Any | None = None,
    *,
    area_epsilon_m2: float = 1.0e-18,
) -> dict[str, Any]:
    """Return topology diagnostics for a triangle mesh used as a closed-volume reference."""

    vertices, faces, indexed_input = _indexed_triangle_mesh(vertices_l, triangles)
    triangle_count = int(faces.shape[0])
    vertex_count = int(vertices.shape[0])
    if triangle_count <= 0 or vertex_count <= 0:
        return {
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "indexed_input": bool(indexed_input),
            "is_edge_watertight": False,
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "degenerate_triangle_count": 0,
            "connected_component_count": 0,
            "euler_characteristic": None,
        }

    tri_vertices = vertices[faces]
    cross = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=-1)
    degenerate = area <= float(area_epsilon_m2)

    undirected_edges: dict[tuple[int, int], int] = {}
    directed_balance: dict[tuple[int, int], int] = {}
    adjacency = [set() for _ in range(vertex_count)]
    for face in faces:
        for left, right in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            edge = (left, right) if left < right else (right, left)
            undirected_edges[edge] = undirected_edges.get(edge, 0) + 1
            directed_balance[edge] = directed_balance.get(edge, 0) + (1 if (left, right) == edge else -1)
            adjacency[left].add(right)
            adjacency[right].add(left)

    boundary_edge_count = sum(1 for count in undirected_edges.values() if count == 1)
    nonmanifold_edge_count = sum(1 for count in undirected_edges.values() if count > 2)
    inconsistent_orientation_edge_count = sum(
        1
        for edge, count in undirected_edges.items()
        if count == 2 and directed_balance.get(edge, 0) != 0
    )
    used_vertices = set(int(value) for value in faces.reshape(-1))
    component_count = _mesh_connected_component_count(adjacency, used_vertices)
    unique_edge_count = int(len(undirected_edges))
    euler_characteristic = int(vertex_count - unique_edge_count + triangle_count)
    degenerate_count = int(np.count_nonzero(degenerate))
    is_edge_watertight = boundary_edge_count == 0 and nonmanifold_edge_count == 0 and degenerate_count == 0
    is_orientable_watertight = is_edge_watertight and inconsistent_orientation_edge_count == 0
    return {
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "indexed_input": bool(indexed_input),
        "unique_edge_count": unique_edge_count,
        "boundary_edge_count": int(boundary_edge_count),
        "nonmanifold_edge_count": int(nonmanifold_edge_count),
        "inconsistent_orientation_edge_count": int(inconsistent_orientation_edge_count),
        "degenerate_triangle_count": degenerate_count,
        "connected_component_count": int(component_count),
        "euler_characteristic": euler_characteristic,
        "is_edge_watertight": bool(is_edge_watertight),
        "is_orientable_watertight": bool(is_orientable_watertight),
        "area_m2_min": float(np.min(area)),
        "area_m2_mean": float(np.mean(area)),
        "area_m2_max": float(np.max(area)),
    }


def weld_duplicate_triangle_vertices(
    vertices_l: Any,
    triangles: Any,
    *,
    decimals: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Merge duplicate-position triangle vertices without changing geometry."""

    vertices = _as_float_array(vertices_l)
    faces = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices_l must have shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"triangles must have shape (T, 3), got {faces.shape}")

    original_vertices = int(vertices.shape[0])
    original_triangles = int(faces.shape[0])
    rounded = np.round(vertices, decimals=int(decimals))
    unique_vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    welded_faces = inverse[faces].astype(np.int64, copy=False)

    valid = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 2] != welded_faces[:, 0])
    )
    welded_faces = welded_faces[valid]
    if welded_faces.size:
        canonical = np.sort(welded_faces, axis=1)
        _, keep = np.unique(canonical, axis=0, return_index=True)
        welded_faces = welded_faces[np.sort(keep)]

    summary = {
        "original_vertex_count": original_vertices,
        "welded_vertex_count": int(unique_vertices.shape[0]),
        "original_triangle_count": original_triangles,
        "welded_triangle_count": int(welded_faces.shape[0]),
        "removed_degenerate_triangle_count": int(original_triangles - int(np.count_nonzero(valid))),
        "removed_duplicate_triangle_count": int(int(np.count_nonzero(valid)) - welded_faces.shape[0]),
    }
    return unique_vertices.astype(np.float32, copy=False), welded_faces.astype(np.int64, copy=False), summary


@dataclass(frozen=True)
class UrdfPressureLayoutSource:
    """Load taxel layout metadata declared on URDF toucher/touch links."""

    urdf_path: str | Path
    base_dir: str | Path | None = None
    require_files: bool = True

    def touch_links(self) -> list[str]:
        return load_touch_links_from_urdf(self.urdf_path)

    def pressure_touch_links(self) -> list[str]:
        return load_pressure_touch_links_from_urdf(self.urdf_path)

    def specs(self) -> list[UrdfPressurePadSpec]:
        return load_pressure_pad_specs_from_urdf(
            self.urdf_path,
            base_dir=self.base_dir,
            require_files=self.require_files,
        )

    def taxel_maps(self, *, backend_like: Any | None = None) -> list[PressureTaxelMap]:
        return load_pressure_taxel_maps_from_urdf(
            self.urdf_path,
            base_dir=self.base_dir,
            require_files=self.require_files,
            backend_like=backend_like,
        )

    def layouts(self, *, backend_like: Any | None = None) -> list[TaxelLayout]:
        return [
            TaxelLayout.from_taxel_map(
                taxel_map,
                sensor_id=sensor_id,
                source="urdf_pressure_layout",
            )
            for sensor_id, taxel_map in enumerate(self.taxel_maps(backend_like=backend_like))
        ]


class PhysxContactSource:
    """Normalize sparse PhysX/Isaac contact records into pressure contact events."""

    def __init__(self, taxel_maps: Iterable[PressureTaxelMap]):
        self.adapter = UrdfPressureContactAdapter(taxel_maps)

    def events_from_records(
        self,
        contacts: Iterable[Mapping[str, Any]],
        *,
        link_poses_w: dict[str, tuple[Any, Any]] | None = None,
    ) -> dict[str, PressureContactEventBatch]:
        return self.adapter.events_by_link(
            (_normalize_contact_record(contact) for contact in contacts),
            link_poses_w=link_poses_w,
        )


def _normalize_contact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    contact = dict(record)
    out: dict[str, Any] = {}
    link = _first_present(contact, "source_link", "link_name", "body_name", "body", "source_body")
    if link is not None:
        out["source_link"] = str(link)

    point_l = _first_present(contact, "contact_point_l", "point_l", "local_point")
    point_w = _first_present(contact, "contact_point_w", "point_w", "position_w", "position")
    normal_l = _first_present(contact, "contact_normal_l", "normal_l", "local_normal")
    normal_w = _first_present(contact, "contact_normal_w", "normal_w", "normal")
    if point_l is not None:
        out["contact_point_l"] = point_l
    elif point_w is not None:
        out["contact_point_w"] = point_w
    if normal_l is not None:
        out["contact_normal_l"] = normal_l
    elif normal_w is not None:
        out["contact_normal_w"] = normal_w

    normal_force = _first_present(contact, "normal_force", "force_n", "normal_force_n")
    force = _first_present(contact, "force")
    if normal_force is None:
        normal_force = _scalar_or_none(force)
    if normal_force is None:
        force_l = _first_present(contact, "force_l", "contact_force_l")
        force_w = _first_present(contact, "force_w", "contact_force_w")
        ambiguous_force = _vector_or_none(force)
        if force_l is None and force_w is None and ambiguous_force is not None:
            if normal_l is not None:
                force_l = ambiguous_force
            elif normal_w is not None:
                force_w = ambiguous_force
        if force_l is not None and normal_l is not None:
            normal_force = _normal_force_component(force_l, normal_l)
        elif force_w is not None and normal_w is not None:
            normal_force = _normal_force_component(force_w, normal_w)
    out["normal_force"] = float(normal_force) if normal_force is not None else 0.0

    shear_l = _first_present(contact, "shear_force_l", "tangent_force_l")
    if shear_l is not None:
        out["shear_force_l"] = shear_l
    return out


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _scalar_or_none(value: Any | None) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def _vector_or_none(value: Any | None) -> Any | None:
    if value is None:
        return None
    return None if _scalar_or_none(value) is not None else value


def _normal_force_component(force: Any, normal: Any) -> float:
    force_vec = np.asarray(force, dtype=np.float32)
    normal_vec = np.asarray(normal, dtype=np.float32)
    norm = float(np.linalg.norm(normal_vec))
    if norm <= 1.0e-12:
        return 0.0
    return max(float(np.dot(force_vec, normal_vec / norm)), 0.0)


def _clean_penetration(value: Any) -> np.ndarray:
    penetration = np.asarray(value, dtype=np.float32)
    return np.where(penetration <= 1.0e-9, 0.0, penetration).astype(np.float32)


def _as_float_array(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _ray_inputs(ray_origins: Any, ray_directions: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    origins = _as_float_array(ray_origins)
    directions = _as_float_array(ray_directions)
    if origins.shape[-1:] != (3,) or directions.shape[-1:] != (3,):
        raise ValueError(f"ray origins/directions must end in 3, got {origins.shape}/{directions.shape}")
    origins, directions = np.broadcast_arrays(origins, directions)
    prefix = tuple(int(v) for v in origins.shape[:-1])
    flat_origins = origins.reshape(-1, 3).astype(np.float32)
    flat_directions = directions.reshape(-1, 3).astype(np.float32)
    norms = np.linalg.norm(flat_directions, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("ray directions must be non-zero")
    return flat_origins, flat_directions / norms, prefix


def _surface_sample_offsets(offsets_l: Any, prefix: tuple[int, ...]) -> np.ndarray:
    offsets = _as_float_array(offsets_l)
    if offsets.shape[-1:] != (3,):
        raise ValueError(f"surface sample offsets must end in 3, got {offsets.shape}")
    if offsets.ndim == 2:
        if offsets.shape[0] <= 0:
            raise ValueError("surface sample offsets must contain at least one sample")
        flat = np.broadcast_to(offsets[None, :, :], (int(np.prod(prefix, dtype=np.int64)), offsets.shape[0], 3))
        return flat.astype(np.float32)
    if offsets.ndim >= 3:
        if offsets.shape[-2] <= 0:
            raise ValueError("surface sample offsets must contain at least one sample")
        try:
            expanded = np.broadcast_to(offsets, (*prefix, offsets.shape[-2], 3))
        except ValueError as exc:
            raise ValueError(
                "surface sample offsets must have shape (K,3) or broadcast to prefix+(K,3); "
                f"got {offsets.shape} for prefix {prefix}"
            ) from exc
        return expanded.reshape(-1, offsets.shape[-2], 3).astype(np.float32)
    raise ValueError(
        "surface sample offsets must have shape (K,3) or prefix+(K,3); "
        f"got {offsets.shape} for prefix {prefix}"
    )


def _broadcast_vec3(value: Any, prefix: tuple[int, ...]) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"expected vector value ending in 3, got {arr.shape}")
    if len(prefix) > 0 and arr.ndim == 2 and arr.shape == (prefix[0], 3):
        arr = arr.reshape((prefix[0],) + (1,) * (len(prefix) - 1) + (3,))
    return np.broadcast_to(arr, (*prefix, 3)).astype(np.float32)


def _indexed_triangle_mesh(vertices_l: Any, triangles: Any | None) -> tuple[np.ndarray, np.ndarray, bool]:
    vertices = _as_float_array(vertices_l)
    if triangles is None:
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            raise ValueError(
                "vertices_l must have shape (T,3,3) when triangles is None, "
                f"got {vertices.shape}"
            )
        flat = vertices.reshape(-1, 3)
        unique_vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
        return unique_vertices.astype(np.float32), inverse.reshape(vertices.shape[0], 3).astype(np.int64), False

    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices_l must have shape (V,3), got {vertices.shape}")
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"triangles must have shape (T,3), got {faces.shape}")
    if faces.size and (np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]):
        raise ValueError("triangles contain vertex indices outside vertices_l")
    return vertices.astype(np.float32), faces.astype(np.int64), True


def _mesh_connected_component_count(adjacency: list[set[int]], used_vertices: set[int]) -> int:
    remaining = set(used_vertices)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _triangle_vertices(vertices_l: Any, triangles: Any | None) -> np.ndarray:
    vertices = _as_float_array(vertices_l)
    if triangles is None:
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            raise ValueError(
                "vertices_l must have shape (T,3,3) when triangles is None, "
                f"got {vertices.shape}"
            )
        tri_vertices = vertices
    else:
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f"vertices_l must have shape (V,3), got {vertices.shape}")
        faces = np.asarray(triangles, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f"triangles must have shape (T,3), got {faces.shape}")
        tri_vertices = vertices[faces]
    if tri_vertices.shape[0] == 0:
        raise ValueError("triangle mesh must contain at least one triangle")
    return np.asarray(tri_vertices, dtype=np.float32)
