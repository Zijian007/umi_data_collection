#!/usr/bin/env bash
# run_kiwi_segment.sh -- KIWI-style offline trajectory for ONE Insta360 X3 4K60 recording (called by `umi-traj`).
#
#   front lens (and optionally rear lens) -> own visual-only ORB-SLAM3 map -> per-frame relocalization in EVERY map of the atlas
#   -> offline GTSAM fusion with the IMU (kiwi_fuse3s.py) -> T_WI trajectory (TUM), metric, gravity-aligned, from frame 0.
#
# usage:
#   run_kiwi_segment.sh --insv <VID_..._NNN.insv> --euroc <dataset>/mav0 --out <dir> [--mode handheld|dual] [--cores 0-3]
#                       [--times <times.csv>] [--lens-video-from <dir of an earlier run>]
#
#   --mode handheld (default): front lens + IMU (hand-held recordings; the rear lens mostly sees the operator)
#   --mode dual              : front + rear lens + IMU (when the rear lens sees the room, e.g. robot-mounted)
#   --cores                  : optional CPU list for taskset (default: no pinning)
#   --times                  : per-frame times (column t_canonical_s) for recordings whose frame timing is irregular
#                              (UNVALIDATED path; `umi-traj` only uses it with --allow-irregular)
#   --lens-video-from        : reuse the lens video(s) of an earlier run of the SAME recording (umi-traj's re-mapping tries)
#   <dataset>/mav0 is written by software/trajectory/x3_extract.py (cam0 timestamps + imu0 of the SAME recording;
#   frame k of the lens videos == dataset frame k).
#
# Environment (set by <deps>/env.sh, which install.sh writes and `umi-traj` sources):
#   KIWI_PY          python with numpy, scipy, gtsam (fusion) -- also used for the lens-video / merge helpers
#   ORB3_DIR         ORB-SLAM3 tree built by orbslam3/build_orbslam3.sh (lib/ + Examples/Monocular-Inertial/x3_mono_slam)
#   ORB3_VOCAB       ORBvoc.txt
#   ORB3_EXTRA_LD    library path for x3_mono_slam (conda env lib + Pangolin)
#   ffmpeg / ffprobe must be on PATH
#   KIWI_CAM_IMU_OFFSET_NS, KIWI_REAR_Q_IC, KIWI_REAR_LEVER   sample-calibration time offset; rear lens Basalt 065
#   KIWI_FRONT_SETTING, KIWI_FUSER   front-lens ORB-SLAM3 settings and fusion script; default = the validated files in this
#                    folder. `umi-traj` points them (and KIWI_CAM_IMU_OFFSET_NS) at a user calibration bundle when one is
#                    installed (software/trajectory/calibration.py)
set -u -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${KIWI_PY:?KIWI_PY not set -- run install.sh and use ./umi-traj (it sources <deps>/env.sh)}"
: "${KIWI_PY_TOOLS:=$KIWI_PY}"
: "${ORB3_DIR:?ORB3_DIR not set -- run install.sh}"
: "${ORB3_VOCAB:=$ORB3_DIR/Vocabulary/ORBvoc.txt}"
: "${ORB3_EXTRA_LD:=}"
: "${KIWI_CAM_IMU_OFFSET_NS:=8442942}"
: "${KIWI_REAR_Q_IC:=0.500794,0.498606,-0.501155,0.499441}"
: "${KIWI_REAR_LEVER:=0.0147,-0.0093,-0.0163}"
: "${KIWI_FRONT_SETTING:=$HERE/config/front_x3_r1440_720.yaml}"
: "${KIWI_FUSER:=$HERE/kiwi_fuse3s.py}"

INSV=""; EUROC=""; OUT=""; MODE=handheld; CORES=""; TIMES=""; REUSE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --insv) INSV=$2; shift 2;; --euroc) EUROC=$2; shift 2;; --out) OUT=$2; shift 2;;
    --mode) MODE=$2; shift 2;; --cores) CORES=$2; shift 2;; --times) TIMES=$2; shift 2;; --lens-video-from) REUSE=$2; shift 2;;
    -h|--help) sed -n 2,33p "$0"; exit 0;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -f "$INSV" ] && [ -f "$EUROC/cam0/data.csv" ] && [ -f "$EUROC/imu0/data.csv" ] && [ -n "$OUT" ] || { echo "need --insv, --euroc <.../mav0>, --out (see --help)"; exit 2; }
case "$MODE" in handheld|dual) ;; *) echo "--mode must be handheld or dual"; exit 2;; esac
[ -x "$ORB3_DIR/Examples/Monocular-Inertial/x3_mono_slam" ] && [ -f "$ORB3_VOCAB" ] || { echo "ORB-SLAM3 not built under $ORB3_DIR (run install.sh)"; exit 2; }
PIN=(); [ -n "$CORES" ] && PIN=(taskset -c "$CORES")
mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd)
NFR=$(($(grep -vc '^#' "$EUROC/cam0/data.csv")))
echo "KIWI_START $(date -u +%FT%TZ) insv=$(basename "$INSV") frames=$NFR mode=$MODE cores=${CORES:-all}" | tee "$OUT/logs/run.txt"
{ echo "kiwi_fuse3s.py $(sha256sum "$KIWI_FUSER" | cut -c1-16) front_setting $(sha256sum "$KIWI_FRONT_SETTING" | cut -c1-16) cam_imu_offset_ns $KIWI_CAM_IMU_OFFSET_NS"; echo "x3_mono_slam $(sha256sum "$ORB3_DIR/Examples/Monocular-Inertial/x3_mono_slam" | cut -c1-16) lib $(sha256sum "$ORB3_DIR/lib/libORB_SLAM3.so" | cut -c1-16)"; } >> "$OUT/logs/run.txt"

# ---- 1. one 720x720 video per lens, frame-aligned with the dataset ----
LENSES="front"; [ "$MODE" = dual ] && LENSES="front rear"
for L in $LENSES; do
  half=right; [ "$L" = rear ] && half=left
  mkdir -p "$OUT/$L"
  if [ -n "$REUSE" ] && [ -s "$REUSE/$L/$L.mp4" ]; then
    ln -sf "$(cd "$REUSE/$L" && pwd)/$L.mp4" "$OUT/$L/$L.mp4"; echo "$L video reused from $REUSE" >> "$OUT/logs/run.txt"; continue; fi
  "$KIWI_PY_TOOLS" "$HERE/make_lens_video.py" "$INSV" "$EUROC" "$OUT/$L/$L.mp4" $half > "$OUT/logs/${L}_video.log" 2>&1 \
    || { echo "LENS_VIDEO_FAILED $L (see logs/${L}_video.log)" | tee -a "$OUT/logs/run.txt"; exit 3; }
done

# ---- 2. ORB-SLAM3: build the lens' own map, then relocalize every frame in every map of the atlas ----
run_orb() {  # <dir> <video> <setting> <mask|none> <map_build|localize:atlas> [map_index]
  local d=$1 video=$2 setting=$3 mask=$4 mode=$5 mi=${6:-0}; mkdir -p "$d"
  local args=(); [ "$mask" != none ] && args+=(--mask_img "$mask")
  case "$mode" in map_build) args+=(--save_map "$d/map.osa" --output_trajectory_csv "$d/kf_traj.csv");; localize:*) args+=(--load_map "${mode#localize:}" --localization_only);; esac
  local to=$(( NFR / 30 * 16 + 300 ))
  ( cd "$ORB3_DIR/Examples/Monocular-Inertial" && ORB_MAP_INDEX=$mi \
    LD_LIBRARY_PATH="$ORB3_DIR/lib:$ORB3_DIR/Thirdparty/DBoW2/lib:$ORB3_DIR/Thirdparty/g2o/lib:$ORB3_EXTRA_LD" \
    nice -n 10 "${PIN[@]}" timeout $to ./x3_mono_slam --vocabulary "$ORB3_VOCAB" --setting "$setting" --input_video "$video" \
      --output_frames_csv "$d/frames.csv" --aruco_dict_id 20 --init_tag_id 13 --init_tag_size 0.024 "${args[@]}" > "$d/log.txt" 2> "$d/stderr.txt" )
  echo "$(basename "$d"): rc=$? $(grep -o 'TRACKED.*' "$d/log.txt")"
}
for L in $LENSES; do
  if [ "$L" = front ]; then SET="$KIWI_FRONT_SETTING"; MASK="${KIWI_FRONT_MASK:-$HERE/config/front_hand_mask.png}"   # KIWI: hands masked while mapping
  else SET="$HERE/config/rear_basalt065_720.yaml"; MASK=none; fi
  run_orb "$OUT/$L/map" "$OUT/$L/$L.mp4" "$SET" "$MASK" map_build | tee -a "$OUT/logs/run.txt"
  if [ "$L" = front ] && [ ! -s "$OUT/$L/map/map.osa" ]; then   # without a front map nothing below can work
    echo "MAP_FAILED front (no map saved; see front/map/log.txt)" | tee -a "$OUT/logs/run.txt"; exit 5; fi
  run_orb "$OUT/$L/loc_m0" "$OUT/$L/$L.mp4" "$SET" none "localize:$OUT/$L/map/map.osa" 0 | tee -a "$OUT/logs/run.txt"   # queries unmasked
  NMAP=$(grep -c "^atlas_map index" "$OUT/$L/loc_m0/log.txt")
  SRCS=(--src "m0:$(grep '^atlas_map index 0 ' "$OUT/$L/loc_m0/log.txt" | awk '{print $7}'):$OUT/$L/loc_m0/frames.csv")
  for ((i=1; i<NMAP; i++)); do
    run_orb "$OUT/$L/loc_m$i" "$OUT/$L/$L.mp4" "$SET" none "localize:$OUT/$L/map/map.osa" $i | tee -a "$OUT/logs/run.txt"
    SRCS+=(--src "m$i:$(grep "^atlas_map index $i " "$OUT/$L/loc_m0/log.txt" | awk '{print $7}'):$OUT/$L/loc_m$i/frames.csv")
  done
  mkdir -p "$OUT/$L/loc"
  "$KIWI_PY_TOOLS" "$HERE/merge_frames.py" --out "$OUT/$L/loc/frames.csv" "${SRCS[@]}" | tee -a "$OUT/logs/run.txt"
done

# ---- 3. offline fusion with the IMU ----
FL="$OUT/front/loc/frames.csv"; RL="$FL"; X=(--front_only)
[ "$MODE" = dual ] && { RL="$OUT/rear/loc/frames.csv"; X=(); }
[ -n "$TIMES" ] && X+=(--times "$TIMES")
nice -n 10 "${PIN[@]}" "$KIWI_PY" "$KIWI_FUSER" "$RL" "$FL" "$EUROC/imu0/data.csv" "$KIWI_CAM_IMU_OFFSET_NS" "$KIWI_REAR_Q_IC" "$OUT/fuse" \
  --lever_prior "$KIWI_REAR_LEVER" --step 30 --zupt --adaptive_sigma --map_residual --node_stride 5 --group_chains 1.25 --split_rot_deg 15 --refine_chains 8 "${X[@]}" \
  > "$OUT/logs/fuse.log" 2>&1
rc=$?; echo "fuse rc=$rc" | tee -a "$OUT/logs/run.txt"
[ $rc -eq 0 ] || { echo "FUSE_FAILED (see logs/fuse.log; NO_USABLE_LENS = no tracking chain passed the IMU self-consistency gate)" | tee -a "$OUT/logs/run.txt"; exit 4; }
echo "KIWI_DONE $(date -u +%FT%TZ) trajectory=$OUT/fuse/traj_tum.txt" | tee -a "$OUT/logs/run.txt"
