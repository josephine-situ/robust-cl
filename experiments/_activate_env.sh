#!/bin/bash
# Sourced by every submit_*.sh to put `robcl_env` on PATH. NOT executable on its
# own -- it needs the caller's shell.
#
# Why this is not just `module load miniforge; conda activate robcl_env`:
# that line is node-dependent and fails a whole array task. On 2026-08-25 a
# 6-task rho sweep had tasks 0-3 succeed and tasks 4-5 (reactor, seeds 7 and 13)
# die on the SAME script with
#   Lmod has detected the following error: The following module(s) are unknown:
#   "miniforge"
# -- the two that ran last landed on nodes whose module tree does not carry that
# name (a different image/arch under a hierarchical Lmod, or a stale spider
# cache; the Lmod message suggests --ignore_cache itself). An array task is hours
# of Gurobi, so it must not be lost to a module name.
#
# Order of attempts, first win stops:
#   1. the conda base cached by a previous SUCCESSFUL task (logs/.conda_base).
#      The filesystem is shared, so once ANY node resolves it every later node
#      skips Lmod entirely. This is what actually closes the failure above.
#   2. `module load` of each name in CONDA_MODULES, plain then --ignore_cache.
#   3. a direct source of conda.sh from CONDA_BASE / $CONDA_EXE / the usual
#      per-user install dirs.
# On success it caches the resolved base for (1) and prints what it used. On
# failure it prints the node, MODULEPATH and which conda-ish modules ARE visible
# there, then returns 1 -- so the log says which node to exclude rather than just
# that a name was unknown. The caller's `set -e` then stops the task before it
# runs python against the wrong interpreter.
#
# Knobs (export before sbatch; --export=ALL propagates them):
#   ROBCL_ENV       conda env name        (default robcl_env)
#   CONDA_MODULES   module names to try   (default "miniforge miniconda anaconda3 anaconda")
#   CONDA_BASE      skip discovery, use this base
#   ROBCL_ENV_CACHE cache file            (default logs/.conda_base)

_robcl_env_name="${ROBCL_ENV:-robcl_env}"
_robcl_cache="${ROBCL_ENV_CACHE:-logs/.conda_base}"
_robcl_modules="${CONDA_MODULES:-miniforge miniconda anaconda3 anaconda}"

# conda's scripts trip over `set -u`, and `module load` of a missing name returns
# nonzero, which `set -e` would turn into a dead task before the next candidate
# is tried. Relax both, restore exactly what the caller had at every exit.
#
# Record the flags from $- rather than `$(set +o)`: inside a command
# substitution bash reports errexit as OFF, so the `set +o` idiom silently
# restores a shell with `set -e` DISABLED -- which would let a failed activation
# fall through into the run instead of stopping it.
_robcl_had_e=0
_robcl_had_u=0
case $- in *e*) _robcl_had_e=1 ;; esac
case $- in *u*) _robcl_had_u=1 ;; esac
set +eu

_robcl_restore() {
    if [[ "${_robcl_had_e}" == "1" ]]; then set -e; fi
    if [[ "${_robcl_had_u}" == "1" ]]; then set -u; fi
    return 0
}

_robcl_log() { echo "[env] $*"; }

# Lmod is a shell function and is NOT always defined in a batch shell.
if ! command -v module >/dev/null 2>&1 && [[ $(type -t module 2>/dev/null) != "function" ]]; then
    for _init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                 "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash"; do
        if [[ -r "${_init}" ]]; then
            _robcl_log "sourcing module init ${_init}"
            . "${_init}"
            break
        fi
    done
fi

# Try one conda base: source its conda.sh and activate. Silent on failure so the
# caller can keep walking candidates.
_robcl_try_base() {
    local base="$1"
    [[ -n "${base}" && -r "${base}/etc/profile.d/conda.sh" ]] || return 1
    . "${base}/etc/profile.d/conda.sh" || return 1
    conda activate "${_robcl_env_name}" 2>/dev/null || return 1
    return 0
}

_robcl_ok=""

# (1) the base a previous successful task resolved.
if [[ -z "${CONDA_BASE:-}" && -s "${_robcl_cache}" ]]; then
    _robcl_cached="$(cat "${_robcl_cache}" 2>/dev/null)"
    if [[ -n "${_robcl_cached}" ]]; then
        if _robcl_try_base "${_robcl_cached}"; then
            _robcl_ok="cached base ${_robcl_cached}"
        else
            _robcl_log "cached base '${_robcl_cached}' did not activate here; falling through"
        fi
    fi
fi

# (2) Lmod, each name plain then with the cache ignored.
if [[ -z "${_robcl_ok}" && -z "${CONDA_BASE:-}" ]]; then
    for _mod in ${_robcl_modules}; do
        for _flag in "" "--ignore_cache"; do
            if module ${_flag} load "${_mod}" 2>/dev/null; then
                if conda activate "${_robcl_env_name}" 2>/dev/null; then
                    _robcl_ok="module ${_mod} ${_flag}"
                    break 2
                fi
                # The module is here but the env is not. Leave it loaded -- the
                # direct-path pass below may still find a usable base.
                _robcl_log "module ${_mod} loaded but '${_robcl_env_name}' would not activate"
            fi
        done
    done
fi

# (3) a conda install by absolute path.
if [[ -z "${_robcl_ok}" ]]; then
    for _base in "${CONDA_BASE:-}" \
                 "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")}" \
                 "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                 "$HOME/mambaforge" /opt/miniforge3 /opt/conda; do
        if _robcl_try_base "${_base}"; then
            _robcl_ok="base ${_base}"
            break
        fi
    done
fi

if [[ -z "${_robcl_ok}" || -z "${CONDA_PREFIX:-}" ]]; then
    _robcl_log "FAILED to activate '${_robcl_env_name}' on $(hostname)"
    _robcl_log "MODULEPATH=${MODULEPATH:-<unset>}"
    _robcl_log "conda-ish modules visible on THIS node:"
    module avail 2>&1 | grep -iE "conda|forge|python" | head -20 || true
    _robcl_log "Fix: resubmit excluding this node (sbatch --exclude=$(hostname) ...),"
    _robcl_log "or, from a login node where it works:"
    _robcl_log "    conda info --base > ${_robcl_cache}"
    _robcl_log "and resubmit -- every task then reads the absolute path instead of Lmod."
    _robcl_restore
    return 1 2>/dev/null || exit 1
fi

# Cache the base for later tasks; best-effort, a read-only FS must not fail a run.
_robcl_base="$(conda info --base 2>/dev/null)"
if [[ -n "${_robcl_base}" && "${_robcl_base}" != "$(cat "${_robcl_cache}" 2>/dev/null)" ]]; then
    mkdir -p "$(dirname "${_robcl_cache}")" 2>/dev/null && \
        printf '%s\n' "${_robcl_base}" > "${_robcl_cache}" 2>/dev/null || true
fi

_robcl_log "activated '${_robcl_env_name}' via ${_robcl_ok}"
_robcl_log "python: $(command -v python)  base: ${_robcl_base:-?}"

_robcl_restore
