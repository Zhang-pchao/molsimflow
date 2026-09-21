#!/usr/bin/env bash
set -eo pipefail
source /etc/profile

[[ $# -eq 4 ]] || {
  echo "usage: run_tpcl_force_step.sh BRANCH_ROOT OUTPUT_ROOT NSTEPS MODE" >&2
  exit 2
}
branch_root=$(readlink -f "$1")
output_root=$2
nsteps=$3
mode=$4
package_root=$(readlink -f "$branch_root/../../..")
source "$branch_root/CASE.env"

cd "$package_root"
sha256sum -c 00_contract/RUNTIME-SHA256SUMS
parent_restart=$(readlink -f "$package_root/$PARENT_RESTART")
model_data=$(readlink -f "$package_root/$MODEL_DATA")
group_file=$(readlink -f "$package_root/$GROUP_FILE")
[[ "$(sha256sum "$parent_restart" | awk '{print $1}')" == "$PARENT_RESTART_SHA256" ]]
[[ "$START_STEP" == 36200000 ]]
[[ "$TIMESTEP_FS" == 0.5 ]]
[[ "$nsteps" =~ ^[0-9]+$ ]]

case "$mode" in
  smoke)
    [[ "$nsteps" -eq 4000 ]]
    fast_steps=4000
    projection_args=(
      --projection-total-steps 200000
      --projection-fast-steps 40000
      --size-ceiling-bytes 3221225472
    )
    ;;
  production)
    [[ "$nsteps" -eq 200000 ]]
    fast_steps=40000
    projection_args=()
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    exit 2
    ;;
esac
slow_steps=$((nsteps - fast_steps))
expected_end=$((START_STEP + nsteps))

if [[ ${PRODUCTION_PRECHECK_ONLY:-0} == 1 ]]; then
  printf 'precheck=PASS\ncase=%s\nbranch=%s\nmode=%s\nstart_step=%s\nnsteps=%s\n' \
    "$CASE_NAME" "$BRANCH_NAME" "$mode" "$START_STEP" "$nsteps"
  exit 0
fi

[[ -n ${SLURM_JOB_ID:-} ]] || { echo "SLURM_JOB_ID is required" >&2; exit 2; }
[[ ! -e "$output_root" ]] || { echo "refusing existing output: $output_root" >&2; exit 2; }
mkdir -p "$output_root"
output_root=$(readlink -f "$output_root")

module purge
module load @@LAMMPS_MODULE@@
module list > "$output_root/MODULES.txt" 2>&1
set -u
unset DP_INTERFACE_PREC DP_CUDA_INFER
export DP_TF32_INFER=0
export DP_TRITON_INFER=0
export OMP_NUM_THREADS=6
export DP_INTRA_OP_PARALLELISM_THREADS=6
export DP_INTER_OP_PARALLELISM_THREADS=1
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv > "$output_root/GPU.txt"
printf 'status=RUNNING\njob_id=%s\nmode=%s\ncase=%s\nbranch=%s\ndirection=%s\n' \
  "$SLURM_JOB_ID" "$mode" "$CASE_NAME" "$BRANCH_NAME" "$DRIVE_DIRECTION" \
  > "$output_root/RUN-RESULT.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "status=FAILED\njob_id=%s\nexit_code=%s\n" \
  "$SLURM_JOB_ID" "$code" > "$output_root/RUN-RESULT.txt"; fi' EXIT

lmp -k on g 1 -sf kk -in "$package_root/01_common/in.tpcl_force_step.lmp" \
  -var MODEL "$package_root/01_common/model/mini500k-compressed.pt2" \
  -var RESTART "$parent_restart" \
  -var GROUP_FILE "$group_file" \
  -var OUTDIR "$output_root" \
  -var FORCE_X "$FORCE_X" -var FORCE_Y "$FORCE_Y" \
  -var NANCHOR "$NANCHOR" -var NDRIVE_O "$NDRIVE_O" \
  -var NWATER "$NWATER" -var NSELECTED "$NSELECTED" \
  -var START_STEP "$START_STEP" \
  -var FAST_STEPS "$fast_steps" -var SLOW_STEPS "$slow_steps" \
  -var MAX_RAW_FORCE "$MAX_RAW_FORCE" -var MAX_O_SPEED "$MAX_O_SPEED" \
  -var MAX_WATER_TEMP "$MAX_WATER_TEMP" \
  -var TABLE_FREQ 20 -var FAST_COORD_FREQ 20 -var SLOW_COORD_FREQ 100 \
  -var DYNAMICS_FREQ 100 -var FULL_FREQ 1000 -var RESTART_FREQ 10000 \
  > "$output_root/lmp.out" 2>&1

last_step=$(awk '$1 ~ /^[0-9]+$/ && NF>5 {last=$1} END{print last+0}' \
  "$output_root/lmp.out")
[[ "$last_step" -eq "$expected_end" ]]
! grep -Eqi 'ERROR:|CUDA.*error|MPI_ABORT|Lost atoms|(^|[[:space:]])(nan|inf)([[:space:]]|$)' \
  "$output_root/lmp.out"
grep -q "ANCHOR_COUNT=$NANCHOR" "$output_root/lmp.out"
grep -q "WATER_ATOM_COUNT=$NWATER" "$output_root/lmp.out"
grep -q "WATER_O_COUNT=$NDRIVE_O" "$output_root/lmp.out"
grep -q "DRIVE_O_COUNT=$NDRIVE_O" "$output_root/lmp.out"
grep -q "TPCL_ANALYSIS_COUNT=$NSELECTED" "$output_root/lmp.out"
for path in \
  final.restart final.data \
  tpcl_coordinates.lammpstrj.zst tpcl_dynamics.lammpstrj.zst \
  full_reference.lammpstrj.zst \
  motion_energy_stress_0p01ps.dat force_sums_0p01ps.dat; do
  [[ -s "$output_root/$path" ]]
done

PYTHONPATH="$package_root/05_postprocess/code_snapshot" \
  @@PYTHON_EXEC@@ -m molsimflow.postprocess.tpcl_force_step_io \
  --output-dir "$output_root" \
  --model-data "$model_data" \
  --start-step "$START_STEP" \
  --total-steps "$nsteps" \
  --fast-steps "$fast_steps" \
  --natoms "$NATOMS" \
  --substrate-atoms "$NSUB" \
  --selected-atoms "$NSELECTED" \
  --expected-water-oxygen "$NDRIVE_O" \
  "${projection_args[@]}" \
  --report "$output_root/VALIDATION.json" \
  > "$output_root/validation.stdout"

find "$output_root" -maxdepth 1 -type f ! -name OUTPUT-SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$output_root/OUTPUT-SHA256SUMS"
printf 'status=PASS\nrun_result=PASS\njob_id=%s\nmode=%s\ncase=%s\nbranch=%s\n' \
  "$SLURM_JOB_ID" "$mode" "$CASE_NAME" "$BRANCH_NAME" > "$output_root/RUN-RESULT.txt"
printf 'direction=%s\nforce_x_eV_per_A=%s\nforce_y_eV_per_A=%s\n' \
  "$DRIVE_DIRECTION" "$FORCE_X" "$FORCE_Y" >> "$output_root/RUN-RESULT.txt"
printf 'parent_job_id=%s\nparent_restart_sha256=%s\nstart_step=%s\nend_step=%s\nend=%s\n' \
  "$PARENT_JOB_ID" "$PARENT_RESTART_SHA256" "$START_STEP" "$last_step" \
  "$(date --iso-8601=seconds)" >> "$output_root/RUN-RESULT.txt"
trap - EXIT
