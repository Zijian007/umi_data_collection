"""一次顺序解码向多个检测器交付原始帧；各检测器独立保存结果。"""

from __future__ import annotations

from software.configuration import operational_settings as settings

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class _DecodedFrame:
    """Per-frame fan-out view with lazy conversion caches."""

    __slots__ = ('_frame', '_arrays', '_crop_grays')

    def __init__(self, frame):
        self._frame = frame
        self._arrays = {}
        self._crop_grays = {}

    def __getattr__(self, name):
        return getattr(self._frame, name)

    def to_ndarray(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in self._arrays:
            self._arrays[key] = self._frame.to_ndarray(*args, **kwargs)
        return self._arrays[key]

    def crop_gray(self, roi, converter):
        key = tuple(roi)
        if key not in self._crop_grays:
            from software.gripper_width.detection.frames import _crop_gray_uncached
            self._crop_grays[key] = _crop_gray_uncached(self._frame, key, converter)
        return self._crop_grays[key]


class FrameConsumer(Protocol):
    def validate(self, streams) -> None: ...
    def consume(self, index: int, frame) -> None: ...
    def finish(self) -> Any: ...


class VideoDecode:
    """只保存当前解码帧；采样消费者可自行保留一帧用于最近 PTS 选择。

    缓存命中的检测器不注册；全部命中时不会打开视频。解码及所有检测器
    完整性检查成功后才调用保存回调，避免截断原片产生完整缓存标记。
    """

    def __init__(self, video: Path):
        self.video = video
        self._consumers: list[tuple[FrameConsumer, Callable[[Any], None] | None]] = []

    def add(self, consumer: FrameConsumer, save: Callable[[Any], None] | None = None) -> None:
        self._consumers.append((consumer, save))

    def run(self) -> list[Any]:
        import av

        if not self._consumers:
            return []
        with av.open(str(self.video)) as container:
            for consumer, _ in self._consumers:
                consumer.validate(container.streams.video)
            stream = container.streams.video[0]
            stream.codec_context.thread_count = settings.values('runtime')['decode_threads']
            for index, frame in enumerate(container.decode(stream)):
                shared = _DecodedFrame(frame)
                for consumer, _ in self._consumers:
                    consumer.consume(index, shared)
        results = [consumer.finish() for consumer, _ in self._consumers]
        for (_, save), result in zip(self._consumers, results):
            if save is not None:
                save(result)
        self._consumers.clear()
        return results
