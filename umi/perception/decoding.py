"""一次顺序解码向多个检测器交付原始帧；各检测器独立保存结果。"""

from __future__ import annotations

from umi.config import operational_settings as settings

from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ScanProgress:
    frames: int
    total: int | None = None
    video_time_s: float | None = None
    observations: int | None = None
    cached: bool = False


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

    def crop_gray(self, roi, converter, crop):
        key = tuple(roi)
        if key not in self._crop_grays:
            self._crop_grays[key] = crop(self._frame, key, converter)
        return self._crop_grays[key]


class FrameConsumer(Protocol):
    def validate(self, streams) -> None: ...
    def consume(self, index: int, frame) -> None: ...
    def finish(self) -> Any: ...


def configure_decoder(stream) -> None:
    """Enable FFmpeg frame threading while keeping decoded frames in display order."""
    stream.codec_context.thread_count = settings.values('runtime')['decode_threads']
    stream.codec_context.thread_type = 'AUTO'


def _deliver(consumers: list[FrameConsumer], index: int, frame) -> None:
    shared = _DecodedFrame(frame)
    for consumer in consumers:
        consumer.consume(index, shared)


class VideoDecode:
    """单次解码，按原片顺序向每个消费者交付帧。

    缓存命中的检测器不注册；全部命中时不会打开视频。解码及所有检测器
    完整性检查成功后才调用保存回调，避免截断原片产生完整缓存标记。
    同组检测器共享图像转换缓存。前后台均有检测时，后台最多积压四帧。
    """

    def __init__(self, video: Path):
        self.video = video
        self._consumers: list[tuple[FrameConsumer, Callable[[Any], None] | None, bool]] = []

    def add(self, consumer: FrameConsumer, save: Callable[[Any], None] | None = None, *,
            background: bool = False) -> None:
        self._consumers.append((consumer, save, background))

    def run(self) -> list[Any]:
        import av

        if not self._consumers:
            return []
        with av.open(str(self.video)) as container:
            for consumer, _, _ in self._consumers:
                consumer.validate(container.streams.video)
            stream = container.streams.video[0]
            configure_decoder(stream)
            frames = enumerate(container.decode(stream))
            foreground = [consumer for consumer, _, background in self._consumers if not background]
            background = [consumer for consumer, _, background in self._consumers if background]
            if foreground and background:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    pending = deque()
                    for index, frame in frames:
                        if len(pending) == 4:
                            pending.popleft().result()
                        pending.append(worker.submit(_deliver, background, index, frame))
                        _deliver(foreground, index, frame)
                    for future in pending:
                        future.result()
            else:
                for index, frame in frames:
                    _deliver(foreground or background, index, frame)
        results = [consumer.finish() for consumer, _, _ in self._consumers]
        for (_, save, _), result in zip(self._consumers, results):
            if save is not None:
                save(result)
        self._consumers.clear()
        return results
