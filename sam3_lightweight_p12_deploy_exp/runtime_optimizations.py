"""不改变模型数学过程的部署期缓存。"""
from types import MethodType
from sam3.model.sam3_image_processor import Sam3Processor


def _is_empty_prompt(prompt):
    """只缓存完全无点、框、mask的几何提示；动态几何提示始终重新编码。"""
    for name in ('point_embeddings', 'box_embeddings'):
        value = getattr(prompt, name, None)
        if value is not None and value.numel():
            return False
    value = getattr(prompt, 'mask_embeddings', None)
    return value is None or value.numel() == 0


class CachedEmptyGeometryProcessor(Sam3Processor):
    """同一张图的多个文本提示共享空几何编码结果。"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._empty_geometry_cache = None
        self.cache_enabled = True
        encoder = model.geometry_encoder
        original = encoder.forward
        self._original_geometry_forward = original

        def cached_forward(_encoder, geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
            if not self.cache_enabled or not _is_empty_prompt(geo_prompt):
                return original(geo_prompt, img_feats, img_sizes, img_pos_embeds)
            if self._empty_geometry_cache is None:
                self._empty_geometry_cache = original(
                    geo_prompt, img_feats, img_sizes, img_pos_embeds
                )
            return self._empty_geometry_cache

        encoder.forward = MethodType(cached_forward, encoder)

    def set_image(self, image, state=None):
        self._empty_geometry_cache = None
        return super().set_image(image, state)
