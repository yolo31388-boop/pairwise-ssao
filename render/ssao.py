import math


def hemisphere_angle(index, total):
    """半球采样角。

    将 ``total`` 个采样均匀铺在半球的一整圈方位角上（0..2π），
    返回第 ``index`` 个采样对应的角度。
    """
    if total <= 0:
        return 0.0
    return 2.0 * math.pi * index / total


def noise_uv(x, y, size):
    """噪声去重复。

    用世界/相机空间坐标对噪声纹理尺寸取模，避免纹理随屏幕平铺
    造成的重复图案；返回值归一化到 [0, 1)。
    """
    if size <= 0:
        return (0.0, 0.0)
    return ((x % size) / size, (y % size) / size)


def linear_depth(z, near, far):
    """深度重建。

    把 [0, 1] 的深度缓冲值按透视投影逆变换为视空间线性深度，
    并显式处理远平面：z=0 对应 near，z=1 对应 far。
    """
    if far <= near:
        return near
    denom = far - z * (far - near)
    if denom == 0:
        return far
    return (near * far) / denom


def ao_radius(world_radius, pixels_per_unit):
    """世界半径。"""
    return world_radius * pixels_per_unit


def normal_weight(dot_value, bias):
    """法线权重。"""
    return min(1.0, max(0.0, dot_value - bias))


def depth_weight(delta, scale):
    """深度权重。"""
    if scale <= 0:
        return 1.0 if delta == 0 else 0.0
    return math.exp(-(delta * delta) / (2.0 * scale * scale))


def upsample_scale(scale, factor):
    """上采样。"""
    return scale * factor


def transparent_occluder(alpha, threshold):
    """透明剔除。"""
    return alpha >= threshold


def distance_falloff(distance, max_distance):
    """距离衰减。"""
    if max_distance <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - distance / max_distance))


def ao_normalize(total, count):
    """AO 归一。"""
    if count <= 0:
        return 0.0
    return total / count


def blur_kernel(radius):
    """双边核。"""
    return 2 * max(0, radius) + 1


def half_res(width):
    """半分辨率。"""
    return width // 2


def sample_count(quality, base):
    """采样数。"""
    return int(quality * base)


def apply_ao(ao, indirect):
    """AO 应用。"""
    return indirect * ao


def camera_ao_key(camera_id, frame):
    """相机 AO 隔离。

    每个相机、每一帧都有独立的 AO 缓冲键，避免多相机分屏时
    互相复用/串扰导致接缝亮度不一致。
    """
    return (camera_id, frame)
