#!/usr/bin/env bash
set -euo pipefail

paper_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
format=${1:-springer-lncs}
length=${2:-full}
source_file="$paper_dir/formats/$format/$length.tex"
output_dir="$paper_dir/build/$format/$length"
input_name="$length.tex"
output_name="$length.pdf"

if [[ ! -f "$source_file" ]]; then
  echo "Paper variant not available: $format/$length" >&2
  exit 2
fi

# Preserve the submitted job name so Tectonic reproduces its PDF ID byte-for-byte.
if [[ "$format/$length" == "springer-lncs/full" ]]; then
  input_name=imitator_e1.tex
  output_name=imitator_e1.pdf
fi

tectonic_bin=${TECTONIC_BIN:-tectonic}
mkdir -p "$output_dir"
(
  cd "$paper_dir/formats/$format"
  SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1783312141} \
    "$tectonic_bin" -Z search-path=template -o "$output_dir" "$input_name"
)

sha256sum "$output_dir/$output_name"
