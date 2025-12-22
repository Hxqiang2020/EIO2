from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

MotionDict = Dict[str, Any]
MotionLoader = Callable[[Union[str, Path]], MotionDict]

REQUIRED_KEYS: Tuple[str, ...] = (
    "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "reset_joint_pos", "reset_joint_vel",
    "reset_root_trans", "reset_root_rot",
    "robot",
)

def load_motion_joblib(file_path: Union[str, Path]) -> MotionDict:
    import joblib
    return joblib.load(str(file_path))


def load_motion_pickle(file_path: Union[str, Path]) -> MotionDict:
    import pickle
    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_motion_npz(file_path: Union[str, Path]) -> MotionDict:
    archive = np.load(str(file_path), allow_pickle=True)
    motion_dict: MotionDict = {key: archive[key] for key in archive.files}

    if "robot" in motion_dict:
        robot_value = motion_dict["robot"]
        if isinstance(robot_value, np.ndarray) and robot_value.ndim == 0:
            motion_dict["robot"] = str(robot_value.item())
        elif isinstance(robot_value, np.ndarray) and robot_value.size == 1:
            motion_dict["robot"] = str(robot_value.reshape(()).item())
    return motion_dict


def is_motion_file(file_path: Path) -> bool:
    suffixes = file_path.suffixes
    return (
        file_path.suffix in (".npz", ".pickle", ".joblib")
        or (".pkl" in suffixes)
    )


def load_motion_auto(file_path: Union[str, Path]) -> MotionDict:
    path = Path(file_path)
    if path.suffix == ".npz":
        return load_motion_npz(path)
    if path.suffix == ".pickle":
        return load_motion_pickle(path)
    if (".pkl" in path.suffixes) or (path.suffix == ".joblib"):
        return load_motion_joblib(path)
    raise ValueError(f"Unsupported motion file: {path.name} (suffixes={path.suffixes})")


def to_float32(array_like: Any) -> np.ndarray:
    arr = np.asarray(array_like)
    return arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr


def require_keys(motion_dict: MotionDict, required_keys: Iterable[str], ctx: str = "") -> None:
    missing = [key for key in required_keys if key not in motion_dict]
    if missing:
        raise KeyError(f"{ctx} missing keys: {missing}")


def require_ndim(name: str, arr: np.ndarray, ndim: int) -> None:
    if arr.ndim != ndim:
        raise ValueError(f"{name}: expected ndim={ndim}, got {arr.ndim}, shape={arr.shape}")


def require_last_dim(name: str, arr: np.ndarray, last_dim: int) -> None:
    if arr.shape[-1] != last_dim:
        raise ValueError(f"{name}: expected last_dim={last_dim}, got {arr.shape[-1]}, shape={arr.shape}")


def quat_normalize(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quat = to_float32(quat)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, eps, None)


def quat_slerp(quat0: np.ndarray, quat1: np.ndarray, weight: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized SLERP.
      quat0, quat1: (..., 4)
      weight: broadcastable to (..., 1) within [0,1]
    """
    quat0 = quat_normalize(quat0, eps)
    quat1 = quat_normalize(quat1, eps)
    weight = to_float32(weight)

    dot = np.sum(quat0 * quat1, axis=-1, keepdims=True)
    quat1 = np.where(dot < 0.0, -quat1, quat1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    near = dot > np.float32(0.9995)
    lerp_quat = quat_normalize(quat0 + weight * (quat1 - quat0), eps)

    theta0 = np.arccos(dot)
    sin0 = np.clip(np.sin(theta0), eps, None)
    theta = theta0 * weight
    w0 = np.sin(theta0 - theta) / sin0
    w1 = np.sin(theta) / sin0
    slerp_quat = quat_normalize(w0 * quat0 + w1 * quat1, eps)

    return np.where(near, lerp_quat, slerp_quat)


def lerp(value0: np.ndarray, value1: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (np.float32(1.0) - weight) * value0 + weight * value1


def calc_frame_blend_const_dt(
    sample_times_sec: np.ndarray,     # (N,)
    motion_lengths_sec: np.ndarray,   # (N,)
    num_frames: np.ndarray,           # (N,) int64
    dt_sec: np.float32,               # scalar
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = sample_times_sec.astype(np.float32, copy=True)
    times[times < 0.0] = 0.0
    times = np.minimum(times, motion_lengths_sec)

    dt_safe = np.float32(max(float(dt_sec), 1e-8))
    frame_coord = times / dt_safe

    frame_idx0 = np.floor(frame_coord).astype(np.int64)
    frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)

    blend = (frame_coord - frame_idx0.astype(np.float32)).astype(np.float32)
    blend = np.clip(blend, 0.0, 1.0)
    return frame_idx0, frame_idx1, blend


def to_1d_ids_times(
    motion_ids: Union[int, Sequence[int], np.ndarray],
    sample_times_sec: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    motion_ids_arr = np.asarray(motion_ids, dtype=np.int64)
    sample_times_arr = to_float32(sample_times_sec)

    if motion_ids_arr.ndim == 0 and sample_times_arr.ndim == 1:
        motion_ids_arr = np.full((sample_times_arr.shape[0],), int(motion_ids_arr), dtype=np.int64)
    if sample_times_arr.ndim == 0 and motion_ids_arr.ndim == 1:
        sample_times_arr = np.full((motion_ids_arr.shape[0],), float(sample_times_arr), dtype=np.float32)

    if motion_ids_arr.ndim != 1 or sample_times_arr.ndim != 1 or motion_ids_arr.shape[0] != sample_times_arr.shape[0]:
        raise ValueError(f"motion_ids and sample_times_sec must be (N,), got {motion_ids_arr.shape} and {sample_times_arr.shape}")

    return motion_ids_arr, sample_times_arr


@dataclass(frozen=True)
class MotionMeta:
    file_path: Path
    motion_key: str
    num_frames: int
    num_links: int
    num_dofs: int


def validate_motion_shapes(motion_dict: MotionDict, motion_key: str) -> Tuple[str, int, int, int]:
    """
    Expected:
      body_pos/body_lin_vel/body_ang_vel: (T, J, 3)
      body_rot: (T, J, 4)
      reset_joint_pos/reset_joint_vel: (T, Dq)
      reset_root_trans: (T, 3)
      reset_root_rot: (T, 4)
    """
    require_keys(motion_dict, REQUIRED_KEYS, ctx=f"[{motion_key}]")

    body_pos = np.asarray(motion_dict["body_pos"])
    body_rot = np.asarray(motion_dict["body_rot"])
    body_lin_vel = np.asarray(motion_dict["body_lin_vel"])
    body_ang_vel = np.asarray(motion_dict["body_ang_vel"])

    reset_joint_pos = np.asarray(motion_dict["reset_joint_pos"])
    reset_joint_vel = np.asarray(motion_dict["reset_joint_vel"])
    reset_root_trans = np.asarray(motion_dict["reset_root_trans"])
    reset_root_rot = np.asarray(motion_dict["reset_root_rot"])

    require_ndim(f"[{motion_key}].body_pos", body_pos, 3)
    require_ndim(f"[{motion_key}].body_rot", body_rot, 3)
    require_ndim(f"[{motion_key}].body_lin_vel", body_lin_vel, 3)
    require_ndim(f"[{motion_key}].body_ang_vel", body_ang_vel, 3)

    require_last_dim(f"[{motion_key}].body_pos", body_pos, 3)
    require_last_dim(f"[{motion_key}].body_rot", body_rot, 4)
    require_last_dim(f"[{motion_key}].body_lin_vel", body_lin_vel, 3)
    require_last_dim(f"[{motion_key}].body_ang_vel", body_ang_vel, 3)

    require_ndim(f"[{motion_key}].reset_joint_pos", reset_joint_pos, 2)
    require_ndim(f"[{motion_key}].reset_joint_vel", reset_joint_vel, 2)
    require_ndim(f"[{motion_key}].reset_root_trans", reset_root_trans, 2)
    require_ndim(f"[{motion_key}].reset_root_rot", reset_root_rot, 2)

    require_last_dim(f"[{motion_key}].reset_root_trans", reset_root_trans, 3)
    require_last_dim(f"[{motion_key}].reset_root_rot", reset_root_rot, 4)

    num_frames = int(body_pos.shape[0])
    num_links = int(body_pos.shape[1])

    if body_rot.shape[:2] != (num_frames, num_links):
        raise ValueError(f"[{motion_key}] body_rot shape {body_rot.shape} mismatches body_pos {body_pos.shape}")
    if body_lin_vel.shape[:2] != (num_frames, num_links):
        raise ValueError(f"[{motion_key}] body_lin_vel shape {body_lin_vel.shape} mismatches (T,J,3)")
    if body_ang_vel.shape[:2] != (num_frames, num_links):
        raise ValueError(f"[{motion_key}] body_ang_vel shape {body_ang_vel.shape} mismatches (T,J,3)")

    if reset_joint_pos.shape[0] != num_frames or reset_joint_vel.shape[0] != num_frames:
        raise ValueError(f"[{motion_key}] reset_joint_* first dim must be T={num_frames}")
    num_dofs = int(reset_joint_pos.shape[1])
    if reset_joint_vel.shape[1] != num_dofs:
        raise ValueError(f"[{motion_key}] reset_joint_pos/vel dof mismatch: {reset_joint_pos.shape} vs {reset_joint_vel.shape}")

    if reset_root_trans.shape[0] != num_frames or reset_root_rot.shape[0] != num_frames:
        raise ValueError(f"[{motion_key}] reset_root_* first dim must be T={num_frames}")

    robot_name = str(motion_dict["robot"])
    return robot_name, num_frames, num_links, num_dofs


def convert_motion_arrays(motion_dict: MotionDict, normalize_quaternions: bool) -> Dict[str, np.ndarray]:
    arrays = {
        "body_pos": to_float32(motion_dict["body_pos"]),
        "body_rot": to_float32(motion_dict["body_rot"]),
        "body_lin_vel": to_float32(motion_dict["body_lin_vel"]),
        "body_ang_vel": to_float32(motion_dict["body_ang_vel"]),
        "reset_joint_pos": to_float32(motion_dict["reset_joint_pos"]),
        "reset_joint_vel": to_float32(motion_dict["reset_joint_vel"]),
        "reset_root_trans": to_float32(motion_dict["reset_root_trans"]),
        "reset_root_rot": to_float32(motion_dict["reset_root_rot"]),
    }
    if normalize_quaternions:
        arrays["body_rot"] = quat_normalize(arrays["body_rot"])
        arrays["reset_root_rot"] = quat_normalize(arrays["reset_root_rot"])
    return arrays

@dataclass
class MotionCache:
    fps: np.float32
    dt_sec: np.float32
    robot_name: str

    motion_keys: np.ndarray          # (M,) object
    num_frames_per_motion: np.ndarray  # (M,) int64
    motion_len_sec: np.ndarray       # (M,) float32
    motion_start_frame: np.ndarray   # (M,) int64

    body_pos: np.ndarray             # (T, J, 3)
    body_rot: np.ndarray             # (T, J, 4)
    body_lin_vel: np.ndarray         # (T, J, 3)
    body_ang_vel: np.ndarray         # (T, J, 3)
    reset_joint_pos: np.ndarray      # (T, 29)
    reset_joint_vel: np.ndarray      # (T, 29)
    reset_root_trans: np.ndarray     # (T, 3)
    reset_root_rot: np.ndarray       # (T, 4)

    num_links: int
    num_dofs: int
    num_motions: int
    total_frames: int


class MotionSampler:
    """
    Pure NumPy motion cache + continuous-time sampling (MotionLib-style).

    Output keys match your naming:
      body_pos/body_rot/body_lin_vel/body_ang_vel/reset_joint_pos/reset_joint_vel/reset_root_trans/reset_root_rot/robot

    If return_debug=True, adds:
      fps/dt/motion_len/num_frames/motion_ids/motion_keys/frame_idx0/1/blend/f0_global/f1_global
    """

    def __init__(self, motion_cache: MotionCache):
        self.motion_cache = motion_cache

    @classmethod
    def build_from_dir(
        cls,
        motion_dir: Union[str, Path],
        fps: float = 50.0,
        file_pattern: str = "*",
        loader: MotionLoader = load_motion_auto,
        normalize_quaternions: bool = True,
        skip_unsupported: bool = True,
    ) -> "MotionSampler":
        motion_dir = Path(motion_dir)
        file_list = sorted(path for path in motion_dir.rglob(file_pattern) if path.is_file())
        if skip_unsupported:
            file_list = [path for path in file_list if is_motion_file(path)]
        if not file_list:
            raise FileNotFoundError(f"No files matched pattern='{file_pattern}' under {motion_dir}")
        return cls.build_from_files(
            file_list,
            fps=fps,
            loader=loader,
            normalize_quaternions=normalize_quaternions,
            skip_unsupported=skip_unsupported,
        )

    @classmethod
    def build_from_files(
        cls,
        motion_files: Sequence[Union[str, Path]],
        fps: float = 50.0,
        loader: MotionLoader = load_motion_auto,
        normalize_quaternions: bool = True,
        skip_unsupported: bool = True,
    ) -> "MotionSampler":
        fps_value = np.float32(fps)
        dt_sec = np.float32(1.0 / float(fps))

        meta_list: List[MotionMeta] = []
        total_frames = 0

        robot_ref: Optional[str] = None
        num_links_ref: Optional[int] = None
        num_dofs_ref: Optional[int] = None

        for file_path_like in motion_files:
            file_path = Path(file_path_like)

            try:
                motion_dict = loader(file_path)
            except ValueError:
                if skip_unsupported:
                    continue
                raise

            robot_name, num_frames, num_links, num_dofs = validate_motion_shapes(motion_dict, motion_key=file_path.stem)
            if num_frames < 2:
                continue

            if robot_ref is None:
                robot_ref = robot_name
            elif robot_name != robot_ref:
                raise ValueError(f"Robot mismatch: {file_path} has '{robot_name}', expected '{robot_ref}'")

            if num_links_ref is None:
                num_links_ref = num_links
            elif num_links != num_links_ref:
                raise ValueError(f"Link count mismatch: {file_path} has J={num_links}, expected {num_links_ref}")

            if num_dofs_ref is None:
                num_dofs_ref = num_dofs
            elif num_dofs != num_dofs_ref:
                raise ValueError(f"DOF mismatch: {file_path} has Dq={num_dofs}, expected {num_dofs_ref}")

            meta_list.append(MotionMeta(file_path=file_path, motion_key=file_path.stem,
                                        num_frames=num_frames, num_links=num_links, num_dofs=num_dofs))
            total_frames += num_frames

        if not meta_list:
            raise RuntimeError("No valid motions loaded (unsupported/too short/shape errors).")

        num_motions = len(meta_list)
        num_links = int(num_links_ref)
        num_dofs = int(num_dofs_ref)
        robot_name = str(robot_ref or "unknown")

        body_pos = np.empty((total_frames, num_links, 3), dtype=np.float32)
        body_rot = np.empty((total_frames, num_links, 4), dtype=np.float32)
        body_lin_vel = np.empty((total_frames, num_links, 3), dtype=np.float32)
        body_ang_vel = np.empty((total_frames, num_links, 3), dtype=np.float32)
        reset_joint_pos = np.empty((total_frames, num_dofs), dtype=np.float32)
        reset_joint_vel = np.empty((total_frames, num_dofs), dtype=np.float32)
        reset_root_trans = np.empty((total_frames, 3), dtype=np.float32)
        reset_root_rot = np.empty((total_frames, 4), dtype=np.float32)

        num_frames_per_motion = np.empty((num_motions,), dtype=np.int64)
        motion_len_sec = np.empty((num_motions,), dtype=np.float32)
        motion_keys = np.empty((num_motions,), dtype=object)

        cursor = 0
        for motion_index, meta in enumerate(meta_list):
            motion_dict = loader(meta.file_path)
            arrays = convert_motion_arrays(motion_dict, normalize_quaternions=normalize_quaternions)

            frame_slice = slice(cursor, cursor + meta.num_frames)
            body_pos[frame_slice] = arrays["body_pos"]
            body_rot[frame_slice] = arrays["body_rot"]
            body_lin_vel[frame_slice] = arrays["body_lin_vel"]
            body_ang_vel[frame_slice] = arrays["body_ang_vel"]
            reset_joint_pos[frame_slice] = arrays["reset_joint_pos"]
            reset_joint_vel[frame_slice] = arrays["reset_joint_vel"]
            reset_root_trans[frame_slice] = arrays["reset_root_trans"]
            reset_root_rot[frame_slice] = arrays["reset_root_rot"]

            num_frames_per_motion[motion_index] = meta.num_frames
            motion_len_sec[motion_index] = dt_sec * np.float32(meta.num_frames - 1)
            motion_keys[motion_index] = meta.motion_key

            cursor += meta.num_frames

        shifted = np.roll(num_frames_per_motion, 1)
        shifted[0] = 0
        motion_start_frame = np.cumsum(shifted).astype(np.int64)

        motion_cache = MotionCache(
            fps=fps_value,
            dt_sec=dt_sec,
            robot_name=robot_name,

            motion_keys=motion_keys,
            num_frames_per_motion=num_frames_per_motion,
            motion_len_sec=motion_len_sec,
            motion_start_frame=motion_start_frame,

            body_pos=body_pos,
            body_rot=body_rot,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            reset_joint_pos=reset_joint_pos,
            reset_joint_vel=reset_joint_vel,
            reset_root_trans=reset_root_trans,
            reset_root_rot=reset_root_rot,

            num_links=num_links,
            num_dofs=num_dofs,
            num_motions=num_motions,
            total_frames=total_frames,
        )
        return cls(motion_cache)

    def save_cache_npz(self, cache_path: Union[str, Path]) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = self.motion_cache

        np.savez_compressed(
            str(cache_path),
            fps=np.asarray(cache.fps),
            dt_sec=np.asarray(cache.dt_sec),
            robot_name=np.asarray(cache.robot_name, dtype=object),

            motion_keys=cache.motion_keys,
            num_frames_per_motion=cache.num_frames_per_motion,
            motion_len_sec=cache.motion_len_sec,
            motion_start_frame=cache.motion_start_frame,

            body_pos=cache.body_pos,
            body_rot=cache.body_rot,
            body_lin_vel=cache.body_lin_vel,
            body_ang_vel=cache.body_ang_vel,
            reset_joint_pos=cache.reset_joint_pos,
            reset_joint_vel=cache.reset_joint_vel,
            reset_root_trans=cache.reset_root_trans,
            reset_root_rot=cache.reset_root_rot,

            num_links=np.asarray(cache.num_links, dtype=np.int64),
            num_dofs=np.asarray(cache.num_dofs, dtype=np.int64),
        )

    @classmethod
    def load_cache_npz(cls, cache_path: Union[str, Path]) -> "MotionSampler":
        archive = np.load(str(cache_path), allow_pickle=True)

        motion_cache = MotionCache(
            fps=np.asarray(archive["fps"]).astype(np.float32).item(),
            dt_sec=np.asarray(archive["dt_sec"]).astype(np.float32).item(),
            robot_name=str(np.asarray(archive["robot_name"]).item()),

            motion_keys=archive["motion_keys"],
            num_frames_per_motion=archive["num_frames_per_motion"].astype(np.int64),
            motion_len_sec=archive["motion_len_sec"].astype(np.float32),
            motion_start_frame=archive["motion_start_frame"].astype(np.int64),

            body_pos=archive["body_pos"].astype(np.float32),
            body_rot=archive["body_rot"].astype(np.float32),
            body_lin_vel=archive["body_lin_vel"].astype(np.float32),
            body_ang_vel=archive["body_ang_vel"].astype(np.float32),
            reset_joint_pos=archive["reset_joint_pos"].astype(np.float32),
            reset_joint_vel=archive["reset_joint_vel"].astype(np.float32),
            reset_root_trans=archive["reset_root_trans"].astype(np.float32),
            reset_root_rot=archive["reset_root_rot"].astype(np.float32),

            num_links=int(np.asarray(archive["num_links"]).astype(np.int64).item()),
            num_dofs=int(np.asarray(archive["num_dofs"]).astype(np.int64).item()),
            num_motions=int(archive["num_frames_per_motion"].shape[0]),
            total_frames=int(archive["body_pos"].shape[0]),
        )
        return cls(motion_cache)

    def sample(
        self,
        motion_ids: Union[int, Sequence[int], np.ndarray],
        sample_times_sec: Union[float, Sequence[float], np.ndarray],
        pos_offset: Optional[np.ndarray] = None,  # (N,3) added to body_pos
        return_debug: bool = False,
    ) -> MotionDict:
        cache = self.motion_cache
        motion_ids_arr, sample_times_arr = to_1d_ids_times(motion_ids, sample_times_sec)

        num_queries = motion_ids_arr.shape[0]
        motion_ids_arr = motion_ids_arr % cache.num_motions

        motion_lengths = cache.motion_len_sec[motion_ids_arr]         # (N,)
        num_frames = cache.num_frames_per_motion[motion_ids_arr]      # (N,)
        start_frames = cache.motion_start_frame[motion_ids_arr]       # (N,)

        frame_idx0, frame_idx1, blend = calc_frame_blend_const_dt(
            sample_times_sec=sample_times_arr,
            motion_lengths_sec=motion_lengths,
            num_frames=num_frames,
            dt_sec=cache.dt_sec,
        )

        global_idx0 = frame_idx0 + start_frames
        global_idx1 = frame_idx1 + start_frames

        weight_body = blend[:, None, None]  # (N,1,1)
        weight_vec = blend[:, None]         # (N,1)

        body_pos0, body_pos1 = cache.body_pos[global_idx0], cache.body_pos[global_idx1]
        body_rot0, body_rot1 = cache.body_rot[global_idx0], cache.body_rot[global_idx1]
        body_lin_vel0, body_lin_vel1 = cache.body_lin_vel[global_idx0], cache.body_lin_vel[global_idx1]
        body_ang_vel0, body_ang_vel1 = cache.body_ang_vel[global_idx0], cache.body_ang_vel[global_idx1]

        reset_joint_pos0, reset_joint_pos1 = cache.reset_joint_pos[global_idx0], cache.reset_joint_pos[global_idx1]
        reset_joint_vel0, reset_joint_vel1 = cache.reset_joint_vel[global_idx0], cache.reset_joint_vel[global_idx1]
        reset_root_trans0, reset_root_trans1 = cache.reset_root_trans[global_idx0], cache.reset_root_trans[global_idx1]
        reset_root_rot0, reset_root_rot1 = cache.reset_root_rot[global_idx0], cache.reset_root_rot[global_idx1]

        body_pos = lerp(body_pos0, body_pos1, weight_body)
        if pos_offset is not None:
            pos_offset_arr = to_float32(pos_offset)
            if pos_offset_arr.shape != (num_queries, 3):
                raise ValueError(f"pos_offset must be (N,3), got {pos_offset_arr.shape}")
            body_pos = body_pos + pos_offset_arr[:, None, :]

        output: MotionDict = {
            # keys aligned to your original naming
            "body_pos": body_pos,
            "body_rot": quat_slerp(body_rot0, body_rot1, weight_body),
            "body_lin_vel": lerp(body_lin_vel0, body_lin_vel1, weight_body),
            "body_ang_vel": lerp(body_ang_vel0, body_ang_vel1, weight_body),

            "reset_joint_pos": lerp(reset_joint_pos0, reset_joint_pos1, weight_vec),
            "reset_joint_vel": lerp(reset_joint_vel0, reset_joint_vel1, weight_vec),
            "reset_root_trans": lerp(reset_root_trans0, reset_root_trans1, weight_vec),
            "reset_root_rot": quat_slerp(reset_root_rot0, reset_root_rot1, weight_vec),

            "robot": cache.robot_name,
        }

        if return_debug:
            output.update({
                "fps": np.float32(cache.fps),
                "dt": np.full((num_queries,), cache.dt_sec, dtype=np.float32),
                "motion_len": motion_lengths.astype(np.float32),
                "num_frames": num_frames.astype(np.int64),

                "motion_ids": motion_ids_arr.astype(np.int64),
                "motion_keys": cache.motion_keys[motion_ids_arr],

                "frame_idx0": frame_idx0.astype(np.int64),
                "frame_idx1": frame_idx1.astype(np.int64),
                "blend": blend.astype(np.float32),

                "f0_global": global_idx0.astype(np.int64),
                "f1_global": global_idx1.astype(np.int64),
            })

        return output

if __name__ == "__main__":
    sampler = MotionSampler.build_from_dir(
        motion_dir="/path/to/motions",
        fps=50.0,
        file_pattern="*",
        loader=load_motion_auto,
        normalize_quaternions=True,
        skip_unsupported=True,
    )
    sampler.save_cache_npz("tmp/motion_cache.npz")

    sampler = MotionSampler.load_cache_npz("tmp/motion_cache.npz")
    motion_ids = np.array([0, 2, 5], dtype=np.int64)
    sample_times_sec = np.array([0.10, 0.33, 1.20], dtype=np.float32)

    sampled = sampler.sample(motion_ids, sample_times_sec, return_debug=True)
    print(sampled["body_pos"].shape)         # (T, J, 3)
    print(sampled["reset_joint_pos"].shape)  # (T, 29)
    print(sampled["motion_keys"])