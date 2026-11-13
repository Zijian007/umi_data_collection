"""Dependency-free local HTTP server for the UMI trajectory viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import argparse
import json
import mimetypes
import os
import re

from software import settings
from software.viewer import media
from software.viewer.cache import root as cache_root

from software.viewer.data import RecordingCatalog
from software.viewer.errors import ViewerDataError
from software.viewer.media import decode_frame, preview_file, preview_status, video_version


STATIC = Path(__file__).resolve().parent / 'static'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'scenes'


def byte_range(header: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or size <= 0 or not any(match.groups()):
        raise ValueError('无效的视频字节区间')
    first, last = match.groups()
    if not first:
        count = int(last)
        if count == 0:
            raise ValueError('无效的视频字节区间')
        return max(0, size - count), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError('视频字节区间越界')
    return start, end


def make_handler(catalog: RecordingCatalog):
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
                super().log_message(message, *args)

        def send_content_headers(self, status, content_type, length, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            if status != 304:
                self.send_header('Content-Length', str(length))
            self.send_header('X-Content-Type-Options', 'nosniff')
            if 'Cache-Control' not in (extra or {}):
                self.send_header('Cache-Control', 'no-cache')
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()

        def json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_content_headers(status, 'application/json; charset=utf-8', len(content))
            if self.command != 'HEAD':
                self.wfile.write(content)

        def file(self, path, video=False, immutable=False):
            with path.open('rb') as source:
                stat = os.fstat(source.fileno())
                size, start, end, status = stat.st_size, 0, stat.st_size - 1, 200
                extra = {}
                content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                if video:
                    etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
                    extra.update({'Accept-Ranges': 'bytes', 'ETag': etag})
                    if immutable:
                        extra['Cache-Control'] = 'private, max-age=86400, immutable'
                    requested = self.headers.get('Range')
                    if not requested and self.headers.get('If-None-Match') == etag:
                        self.send_content_headers(304, content_type, 0, extra)
                        return
                    if requested and self.headers.get('If-Range', etag) == etag:
                        try:
                            start, end = byte_range(requested, size)
                        except ValueError:
                            self.send_content_headers(416, content_type, 0, {'Content-Range': f'bytes */{size}'})
                            return
                        status = 206
                        extra['Content-Range'] = f'bytes {start}-{end}/{size}'
                if path.suffix == '.mjs':
                    content_type = 'text/javascript'
                self.send_content_headers(status, content_type, max(0, end - start + 1), extra)
                if self.command == 'HEAD':
                    return
                source.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = source.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                route = unquote(urlsplit(self.path).path)
                if route == '/api/recordings':
                    self.json(catalog.scan())
                elif match := re.fullmatch(r'/api/recordings/([0-9a-f]{20})(/video|/frame|/map|/preview|/preview/video)?', route):
                    key, media = match.groups()
                    if media == '/map':
                        self.json(catalog.map_data(key))
                    elif media:
                        path = catalog.video_path(key)
                        if path is None:
                            self.json({'error': '未找到工作镜头视频'}, 404)
                        elif media == '/preview':
                            status = preview_status(path)
                            if status['status'] == 'ready':
                                status['url'] = f'/api/recordings/{key}/preview/video?v={status["version"]}'
                            self.json(status)
                        elif media == '/preview/video':
                            query = parse_qs(urlsplit(self.path).query)
                            self.file(preview_file(path, query.get('v', [''])[0]), video=True, immutable=True)
                        elif media == '/frame':
                            query = parse_qs(urlsplit(self.path).query)
                            seconds = float(query.get('t', ['0'])[0])
                            version = query.get('v', [None])[0]
                            if version is not None and version != video_version(path):
                                raise ViewerDataError('原视频已更新，请重新选择录像。')
                            image = decode_frame(path, seconds)
                            headers = {'Cache-Control': 'private, max-age=86400, immutable'} if version else None
                            self.send_content_headers(200, 'image/jpeg', len(image), headers)
                            if self.command != 'HEAD':
                                self.wfile.write(image)
                        else:
                            self.file(path, video=True)
                    else:
                        self.json(catalog.load(key))
                else:
                    relative = 'index.html' if route == '/' else route.lstrip('/')
                    path = (STATIC / relative).resolve()
                    if not path.is_relative_to(STATIC) or not path.is_file():
                        self.json({'error': '页面不存在'}, 404)
                    else:
                        self.file(path)
            except (ViewerDataError, ValueError) as error:
                self.json({'error': str(error)}, 400)
            except OSError as error:
                if not isinstance(error, (BrokenPipeError, ConnectionResetError)):
                    self.json({'error': f'无法读取文件：{error}'}, 404)

    return ViewerHandler


def parse_port(value: str) -> int:
    port = int(value)
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError('端口必须在 0–65535 之间；0 表示自动选择空闲端口')
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-view', description='本地 UMI 三维轨迹、姿态和视频同步查看器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  umi-view\n  umi-view scenes/test_shelf --port 9092\n'
               '  umi-view /path/to/take_001/results\n启动后按终端地址访问网页，Ctrl+C 停止。\n'
               '完整参数参考：docs/命令行参数.md。')
    parser.add_argument('root', nargs='?', type=Path, default=DEFAULT_ROOT, metavar='数据目录',
                        help='场景/批次/任务目录或单个结果目录；默认扫描本项目的 scenes/')
    parser.add_argument('--host', default=None, help='监听地址，默认 viewer.yaml 的 host（127.0.0.1）；远程访问可指定 0.0.0.0')
    parser.add_argument('--port', type=parse_port, default=None,
                        help='监听端口，默认 viewer.yaml 的 port（9092）；范围 0–65535，0 自动选择空闲端口')
    parser.add_argument('--show-config', action='store_true', help='只输出生效启动配置 JSON，不扫描结果、不监听端口；查看器不读取算法 YAML')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        defaults = settings.current('viewer')
        export_budget = settings.current('runtime')
        explicit = {key: getattr(args, key) for key in ('host', 'port') if getattr(args, key) is not None}
        configured = settings.load('viewer', environment={settings.SNAPSHOT_ENV: json.dumps({'viewer': defaults.report()})}, overrides=explicit)
        args.host, args.port = configured.values['host'], configured.values['port']
        if args.show_config:
            print(json.dumps(dict(program='umi-view', operation='serve', scope='configuration_only',
                precedence=configured.report()['precedence'], defaults=dict(root=str(DEFAULT_ROOT), **defaults.values),
                effective=dict(root=str(args.root.expanduser().resolve()), **configured.values),
                configuration=configured.report(),
                map_export=dict(timeout_s=export_budget.values['map_export_timeout_s'],
                                origin=export_budget.origins['map_export_timeout_s']),
                cache=dict(path=str(cache_root()), origin='environment:UMI_VIEW_CACHE' if os.environ.get('UMI_VIEW_CACHE')
                           else 'XDG_CACHE_HOME/umi-view or ~/.cache/umi-view')),
                ensure_ascii=False, indent=2))
            return 0
        catalog = RecordingCatalog(args.root)
        catalog.scan()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    except (OSError, ValueError) as error:
        parser.exit(2, f'无法启动 UMI Trace：{error}\n')
    print(f'UMI Trace · http://{args.host if args.host != "0.0.0.0" else "127.0.0.1"}:{server.server_port}', flush=True)
    print(f'数据目录：{catalog.root}\n配置：{", ".join(configured.sources)}；'
          f'解码并发 {configured.values["decode_concurrency"]}，预览并发 {configured.values["preview_workers"]}，'
          f'每图最多 {configured.values["map_max_points"]} 点；缓存：{cache_root()}\n按 Ctrl+C 停止。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        media.shutdown()
    return 0
