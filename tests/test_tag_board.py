"""PDF 海报布局的几何、身份与坏资产检查，不靠猜测 ID 生成运行时角点。"""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from software import provenance
from software.vision.tag_board import DEFAULT_LAYOUT, load


class TagBoardTests(unittest.TestCase):
    def test_pdf_asset_dimensions_corner_order_and_source(self):
        board = load()
        self.assertEqual(board.ids, tuple(range(12, 89)))
        self.assertEqual((board.dictionary, board.border_bits), ('DICT_4X4_100', 1))
        self.assertEqual(board.geometry_status, 'pdf_design')
        np.testing.assert_allclose(board.corners_m[12], [[.06,.05,0],[.16,.05,0],[.16,.15,0],[.06,.15,0]])
        np.testing.assert_allclose(board.corners_m[88], [[.84,1.35,0],[.94,1.35,0],[.94,1.45,0],[.84,1.45,0]])
        self.assertEqual(provenance.file_hash(DEFAULT_LAYOUT.parent / board.source_pdf['path']), board.source_pdf['sha256'])

    def test_pdf_render_decodes_all_77_ids_in_declared_positions_and_orientations(self):
        renderer = shutil.which('pdftoppm')
        if renderer is None:
            self.skipTest('PDF rasterizer not installed')
        board = load()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'poster'
            subprocess.run([renderer, '-scale-to', '2400', '-png', '-singlefile',
                            str(DEFAULT_LAYOUT.parent / board.source_pdf['path']), str(output)], check=True,
                           capture_output=True, timeout=30)
            image = cv2.imread(str(output.with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
        parameters = cv2.aruco.DetectorParameters()
        parameters.markerBorderBits = board.border_bits
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), parameters)
        corners, ids, _ = detector.detectMarkers(image)
        self.assertIsNotNone(ids)
        self.assertEqual(sorted(ids.ravel().tolist()), list(board.ids))
        scale = np.array([image.shape[1], image.shape[0]]) / board.sheet_size_m
        for tag_id, detected in zip(ids.ravel().tolist(), corners):
            np.testing.assert_allclose(detected.reshape(4,2), board.corners_m[tag_id][:,:2]*scale, atol=2., rtol=0)

    def test_invalid_layouts_are_rejected(self):
        original = json.loads(DEFAULT_LAYOUT.read_text())
        cases = []
        for key, value in [('units','mm'), ('dictionary','DICT_4X4_50'), ('border_bits',True),
                           ('sheet_size_m',[1.,float('nan')]), ('corner_order','unknown')]:
            doc = copy.deepcopy(original); doc[key] = value; cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][1]['id']=12;cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'][1][0]+=.01;cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'].reverse();cases.append(doc)
        doc=copy.deepcopy(original);doc['markers'][0]['corners_m'][0][2]=.001;cases.append(doc)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'layout.json'
            for i, doc in enumerate(cases):
                with self.subTest(i=i):
                    path.write_text(json.dumps(doc))
                    with self.assertRaises(ValueError): load(path)

    def test_geometry_identity_ignores_marker_serialization_order_but_tracks_dimensions(self):
        doc=json.loads(DEFAULT_LAYOUT.read_text())
        original=load()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'layout.json'
            doc['markers'].reverse()
            path.write_text(json.dumps(doc))
            self.assertEqual(load(path).geometry_hash,original.geometry_hash)
            for marker in doc['markers']:
                marker['corners_m']=(np.asarray(marker['corners_m'])*.99).tolist()
            path.write_text(json.dumps(doc))
            self.assertNotEqual(load(path).geometry_hash,original.geometry_hash)
