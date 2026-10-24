"""Dependency-free local HTTP server for the UMI trajectory viewer."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from software.viewer.data import RecordingCatalog, ViewerDataError
from software.viewer.media import decode_frame

STATIC = Path(__file__).resolve().parent / 'static'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'recordings'


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
            self.send_header('Content-Length', str(length))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-cache')
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()

        def json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_content_headers(status, 'application/json; charset=utf-8', len(content))
            if self.command != 'HEAD':
                self.wfile.write(content)

        def file(self, path, video=False):
            with path.open('rb') as source:
                stat = os.fstat(source.fileno())
                size, start, end, status = stat.st_size, 0, stat.st_size - 1, 200
                extra = {}
                if video:
                    etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
                    extra.update({'Accept-Ranges': 'bytes', 'ETag': etag})
                    requested = self.headers.get('Range')
                    if requested and self.headers.get('If-Range', etag) == etag:
                        try:
                            start, end = byte_range(requested, size)
                        except ValueError:
                            self.send_content_headers(416, 'video/mp4', 0, {'Content-Range': f'bytes */{size}'})
                            return
                        status = 206
                        extra['Content-Range'] = f'bytes {start}-{end}/{size}'
                content_type = 'video/mp4' if video else mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
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
                elif match := re.fullmatch(r'/api/recordings/([0-9a-f]{20})(/video|/frame)?', route):
                    key, media = match.groups()
                    if media:
                        path = catalog.video_path(key)
                        if path is None:
                            self.json({'error': '未找到前镜头视频'}, 404)
                        elif media == '/frame':
                            query = parse_qs(urlsplit(self.path).query)
                            seconds = float(query.get('t', ['0'])[0])
                            image = decode_frame(path, seconds)
                            self.send_content_headers(200, 'image/jpeg', len(image))
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog='umi-view', description='本地 UMI 三维轨迹、姿态和视频同步查看器')
    parser.add_argument('root', nargs='?', type=Path, default=DEFAULT_ROOT, help='录像总目录或单个轨迹结果目录')
    parser.add_argument('--host', default='127.0.0.1', help='默认仅本机访问；远程访问可指定 0.0.0.0')
    parser.add_argument('--port', type=int, default=9091)
    args = parser.parse_args(argv)
    try:
        catalog = RecordingCatalog(args.root)
        catalog.scan()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    except (OSError, ValueError) as error:
        parser.exit(2, f'无法启动 UMI Trace：{error}\n')
    print(f'UMI Trace · http://{args.host if args.host != "0.0.0.0" else "127.0.0.1"}:{server.server_port}', flush=True)
    print(f'数据目录：{catalog.root}\n按 Ctrl+C 停止。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
