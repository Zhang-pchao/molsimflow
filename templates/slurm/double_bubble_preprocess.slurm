#!/bin/bash
#SBATCH --job-name=msf-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"

if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
  # The setup script is intentionally external so cluster-specific module,
  # conda, account, and PATH choices stay outside the public template.
  source "$ENV_SETUP_SCRIPT"
fi

if [[ -n "${MOLSIMFLOW_SOURCE_DIR:-}" ]]; then
  export PYTHONPATH="${MOLSIMFLOW_SOURCE_DIR}/src:${PYTHONPATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR}"
STRUCTURE_MODE="${STRUCTURE_MODE:-prebuilt}"
RUN_PACKMOL="${RUN_PACKMOL:-1}"
PACKMOL_COMMAND="${PACKMOL_COMMAND:-packmol}"

GAS_RADIUS_1="${GAS_RADIUS_1:-19}"
GAS_RADIUS_2="${GAS_RADIUS_2:-14}"
BUBBLE_SPACING="${BUBBLE_SPACING:-50}"
TARGET_PH="${TARGET_PH:-13.4}"
BUBBLE_SHAPE="${BUBBLE_SHAPE:-sphere}"
BUFFER="${BUFFER:-2.2}"
INTERLAYER_THICKNESS="${INTERLAYER_THICKNESS:-2.0}"

run_molsimflow() {
  "$PYTHON_BIN" -m molsimflow.cli "$@"
}

append_common_builder_args() {
  local -n args_ref=$1
  args_ref+=(--output-dir "$OUTPUT_DIR")
  args_ref+=(--gas-radii "$GAS_RADIUS_1" "$GAS_RADIUS_2")
  args_ref+=(--bubble-spacing "$BUBBLE_SPACING")
  args_ref+=(--bubble-shape "$BUBBLE_SHAPE")
  args_ref+=(--buffer "$BUFFER")
  args_ref+=(--interlayer-thickness "$INTERLAYER_THICKNESS")
  args_ref+=(--output-system-suffix "${OUTPUT_SYSTEM_SUFFIX:-slab}")
  args_ref+=(--packmol-tolerance "${PACKMOL_TOLERANCE:-2.4}")
  args_ref+=(--packmol-seed "${PACKMOL_SEED:--1}")

  if [[ "${NO_PH:-0}" == "1" ]]; then
    args_ref+=(--no-ph)
  else
    args_ref+=(--target-ph "$TARGET_PH")
  fi

  if [[ "${NO_INTERLAYER:-0}" == "1" ]]; then
    args_ref+=(--no-interlayer)
  fi
  if [[ -n "${WATER_HEIGHT_RADIUS:-}" ]]; then
    args_ref+=(--water-height-radius "$WATER_HEIGHT_RADIUS")
  fi
  if [[ -n "${BUBBLE_CENTER_Z_RADIUS:-}" ]]; then
    args_ref+=(--bubble-center-z-radius "$BUBBLE_CENTER_Z_RADIUS")
  fi
  if [[ -n "${MOLECULE_DIR:-}" ]]; then
    args_ref+=(--molecule-dir "$MOLECULE_DIR")
  fi
  if [[ -n "${WATER_XYZ:-}" ]]; then
    args_ref+=(--water-xyz "$WATER_XYZ")
  fi
  if [[ -n "${N2_XYZ:-}" ]]; then
    args_ref+=(--n2-xyz "$N2_XYZ")
  fi
  if [[ -n "${CATION_XYZ:-}" ]]; then
    args_ref+=(--cation-xyz "$CATION_XYZ")
  fi
  if [[ -n "${ANION_XYZ:-}" ]]; then
    args_ref+=(--anion-xyz "$ANION_XYZ")
  fi
  if [[ "$RUN_PACKMOL" == "1" ]]; then
    args_ref+=(--run-packmol --packmol-command "$PACKMOL_COMMAND")
    if [[ -n "${PACKMOL_LOG:-}" ]]; then
      args_ref+=(--packmol-log "$PACKMOL_LOG")
    fi
  fi
}

generate_structure_inputs() {
  if [[ "${GENERATE_STRUCTURE_INPUTS:-1}" != "1" ]]; then
    return 0
  fi

  case "$STRUCTURE_MODE" in
    prebuilt)
      : "${INTERFACE_STRUCTURE:?Set INTERFACE_STRUCTURE for STRUCTURE_MODE=prebuilt}"
      builder_args=(structure slab-double-bubble --interface-structure "$INTERFACE_STRUCTURE")
      append_common_builder_args builder_args
      builder_args+=(--bottom-layer-index "${BOTTOM_LAYER_INDEX:-0}")
      builder_args+=(--layer-tolerance "${LAYER_TOLERANCE:-0.1}")
      builder_args+=(--slab-name "${SLAB_NAME:-interface_slab}")
      if [[ -n "${BOTTOM_REFERENCE_Z:-}" ]]; then
        builder_args+=(--bottom-reference-z "$BOTTOM_REFERENCE_Z")
      fi
      if [[ -n "${TOP_REFERENCE_Z:-}" ]]; then
        builder_args+=(--top-reference-z "$TOP_REFERENCE_Z")
      fi
      run_molsimflow "${builder_args[@]}"
      ;;
    tio2)
      : "${BULK_STRUCTURE:?Set BULK_STRUCTURE for STRUCTURE_MODE=tio2}"
      builder_args=(structure tio2-double-bubble --bulk-structure "$BULK_STRUCTURE")
      append_common_builder_args builder_args
      builder_args+=(--miller-index ${MILLER_INDEX:-1 0 1})
      builder_args+=(--layers "${LAYERS:-6}")
      builder_args+=(--vacuum "${VACUUM:-100}")
      builder_args+=(--repeat ${REPEAT:-11 20 1})
      builder_args+=(--bottom-layer-index "${BOTTOM_LAYER_INDEX:-4}")
      builder_args+=(--slab-slice-margin "${SLAB_SLICE_MARGIN:-0.5}")
      builder_args+=(--layer-tolerance "${LAYER_TOLERANCE:-0.1}")
      builder_args+=(--slab-name "${SLAB_NAME:-tio2_interface_101}")
      run_molsimflow "${builder_args[@]}"
      ;;
    existing)
      ;;
    *)
      echo "Unsupported STRUCTURE_MODE: $STRUCTURE_MODE" >&2
      exit 2
      ;;
  esac
}

infer_packmol_output() {
  local input_file=$1
  awk 'tolower($1)=="output" {print $2; exit}' "$input_file"
}

run_existing_packmol_if_needed() {
  if [[ "$STRUCTURE_MODE" != "existing" || "$RUN_PACKMOL" != "1" ]]; then
    return 0
  fi
  local input_file="${PACKMOL_INPUT:-$OUTPUT_DIR/packmol.in}"
  local log_file="${PACKMOL_LOG:-$OUTPUT_DIR/packmol.out}"
  if [[ ! -s "$input_file" ]]; then
    echo "PACKMOL input not found: $input_file" >&2
    exit 2
  fi
  local packed_xyz
  packed_xyz=$(infer_packmol_output "$input_file")
  if [[ -z "$packed_xyz" ]]; then
    echo "Could not infer PACKMOL output from $input_file" >&2
    exit 2
  fi
  if [[ -s "$OUTPUT_DIR/$packed_xyz" ]]; then
    echo "Packed XYZ exists; skipping Packmol: $OUTPUT_DIR/$packed_xyz"
    return 0
  fi
  (cd "$OUTPUT_DIR" && $PACKMOL_COMMAND < "$input_file" > "$log_file")
}

convert_outputs() {
  if [[ "${RUN_CONVERSION:-1}" != "1" ]]; then
    return 0
  fi
  local input_file="${PACKMOL_INPUT:-$OUTPUT_DIR/packmol.in}"
  local packed_xyz="${PACKED_XYZ:-$(infer_packmol_output "$input_file")}"
  if [[ -z "$packed_xyz" ]]; then
    echo "Could not infer packed XYZ filename from $input_file" >&2
    exit 2
  fi
  local packed_xyz_path="$packed_xyz"
  if [[ "$packed_xyz_path" != /* ]]; then
    packed_xyz_path="$OUTPUT_DIR/$packed_xyz_path"
  fi
  if [[ ! -s "$packed_xyz_path" ]]; then
    echo "Packed XYZ not found: $packed_xyz_path" >&2
    exit 2
  fi

  local poscar_path="${POSCAR_PATH:-$OUTPUT_DIR/POSCAR}"
  local extxyz_path="${EXTXYZ_OUTPUT:-$OUTPUT_DIR/model.xyz}"
  local lammps_data_path="${LAMMPS_DATA_OUTPUT:-$OUTPUT_DIR/model_atomic.data}"

  run_molsimflow structure add-extxyz-pbc \
    --poscar "$poscar_path" \
    --xyz "$packed_xyz_path" \
    --output "$extxyz_path" \
    --z-min-padding "${Z_MIN_PADDING:-5.0}"

  run_molsimflow structure extxyz-to-lammps-data \
    --xyz "$extxyz_path" \
    --output "$lammps_data_path"
}

generate_structure_inputs
run_existing_packmol_if_needed
convert_outputs

echo "molsimflow double-bubble preprocessing completed"
