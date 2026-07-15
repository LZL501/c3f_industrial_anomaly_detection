#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-data}
DTD_URL=${DTD_URL:-https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz}
DTD_SIZE=${DTD_SIZE:-625239812}
DTD_PARTS=${DTD_PARTS:-16}
DTD_PROXY=${DTD_PROXY:-}

ARCHIVE=dtd-r1.0.1.tar.gz
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

image_count() {
  if [ -d dtd/images ]; then
    find dtd/images -type f | wc -l
  else
    echo 0
  fi
}

if [ "$(image_count)" -ge 5600 ]; then
  echo "DTD_READY=${BASE_DIR}/dtd/images"
  echo "IMAGE_COUNT=$(image_count)"
  exit 0
fi

curl_args=(--fail --location --retry 5 --retry-delay 10 --connect-timeout 30)
if [ -n "${DTD_PROXY}" ]; then
  curl_args+=(--proxy "${DTD_PROXY}")
fi

if [ -f "${ARCHIVE}" ] && [ "$(stat -c%s "${ARCHIVE}")" -eq "${DTD_SIZE}" ]; then
  echo "archive exists: ${BASE_DIR}/${ARCHIVE}"
else
  rm -f "${ARCHIVE}.part."*
  pids=()
  for ((i = 0; i < DTD_PARTS; i++)); do
    start=$((i * DTD_SIZE / DTD_PARTS))
    end=$((((i + 1) * DTD_SIZE / DTD_PARTS) - 1))
    if [ "${i}" -eq $((DTD_PARTS - 1)) ]; then
      end=$((DTD_SIZE - 1))
    fi
    part=$(printf "%s.part.%02d" "${ARCHIVE}" "${i}")
    echo "download part ${i}: bytes ${start}-${end}"
    curl "${curl_args[@]}" --range "${start}-${end}" --output "${part}" "${DTD_URL}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [ "${failed}" -ne 0 ]; then
    echo "DTD download failed" >&2
    exit 1
  fi

  for ((i = 0; i < DTD_PARTS; i++)); do
    start=$((i * DTD_SIZE / DTD_PARTS))
    end=$((((i + 1) * DTD_SIZE / DTD_PARTS) - 1))
    if [ "${i}" -eq $((DTD_PARTS - 1)) ]; then
      end=$((DTD_SIZE - 1))
    fi
    expected=$((end - start + 1))
    part=$(printf "%s.part.%02d" "${ARCHIVE}" "${i}")
    actual=$(stat -c%s "${part}")
    if [ "${actual}" -ne "${expected}" ]; then
      echo "bad part size for ${part}: ${actual}/${expected}" >&2
      exit 1
    fi
  done

  tmp="${ARCHIVE}.tmp"
  : > "${tmp}"
  for ((i = 0; i < DTD_PARTS; i++)); do
    part=$(printf "%s.part.%02d" "${ARCHIVE}" "${i}")
    cat "${part}" >> "${tmp}"
  done
  mv "${tmp}" "${ARCHIVE}"
fi

tar -tzf "${ARCHIVE}" >/dev/null
tar -xzf "${ARCHIVE}"
count=$(image_count)
echo "IMAGE_COUNT=${count}" | tee dtd/image_count.txt
if [ "${count}" -lt 5600 ]; then
  echo "unexpected DTD image count: ${count}" >&2
  exit 1
fi
echo "DTD_READY=${BASE_DIR}/dtd/images"
