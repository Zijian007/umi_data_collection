"""一次顺序解码向多个检测器交付原始帧；各检测器独立保存结果。"""

from __future__ import annotations

from software import settings

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


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
                for consumer, _ in self._consumers:
                    consumer.consume(index, frame)
        results = [consumer.finish() for consumer, _ in self._consumers]
        for (_, save), result in zip(self._consumers, results):
            if save is not None:
                save(result)
        self._consumers.clear()
        return results
