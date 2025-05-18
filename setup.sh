#!/bin/bash
# creates the following structure
# ./
#  ├─ [problem_code]
#  |  ├─ [problem_code].cpp
#  |  ├─ [problem_code].in (with -wf)
#  |  ├─ [problem_code].out (with -wf)

TEMPLATE_SRC="$HOME/main/cpp/template.cpp"

usage() {
  echo "Usage: $0 <problem_code> [-wf]"
  echo "  <problem_code>: Problem code in CF or problem name in other OJ."
  echo "  -wf: with explicit in and out files"
  exit 1
}

create_folder () {
  local problem_code="$1"
  if [ -d "$problem_code" ]; then
    echo "Error: Directory '$problem_code' already exists."
    return 1
  fi

  if mkdir "$problem_code"; then
    if cd "$problem_code"; then
      if cat "$TEMPLATE_SRC" > "${problem_code}.cpp"; then
        return 0
      else
        echo "Error: Failed to copy template."
        return 1
      fi
    else
      echo "Error: Failed to change directory to '$problem_code'."
      return 1
    fi
  else
    echo "Error: Failed to create directory '$problem_code'."
    return 1
  fi
}

handle_file_flag () {
  local problem_code="$1"

  touch "${problem_code}.in" || { echo "Error: Failed to create ${problem_code}.in"; return 1; }
  touch "${problem_code}.out" || { echo "Error: Failed to create ${problem_code}.out"; return 1; }

  # Uncomment everything from the template and replace
  # .in and .out with problem_code.in and problem_code.out
  # for old USACO submissions
  sed -i 's#// ##g' "${problem_code}.cpp" || { echo "Error in sed"; }
  sed -i "s/\.in/${problem_code}.in/g" "${problem_code}.cpp" || { echo "Error in sed"; }
  sed -i "s/\.out/${problem_code}.out/g" "${problem_code}.cpp" || { echo "Error in sed"; }
  return 0
}

if [ ! -f "$TEMPLATE_SRC" ]; then
  echo "Error: Template file not found at '$TEMPLATE_SRC'."
  exit 1
fi

if [ -z "$1" ]; then
  usage
fi

PROBLEM_CODE="$1"
FLAG="$2"

if [ -z "$FLAG" ]; then
  create_folder "$PROBLEM_CODE" || exit 1
elif [ "$FLAG" == "-wf" ]; then
  # -wf is (with file IO)
  if create_folder "$PROBLEM_CODE"; then
    handle_file_flag "$PROBLEM_CODE"
  else
    exit 1
  fi
else
  echo "Invalid second argument: '$FLAG'"
  usage
fi