#!/bin/bash
# Creates the following structure:
# ./
#  ├─ [problem_code]/
#  |  ├─ [problem_code].cpp
#  |  ├─ [io_name].in (with -wf <io_name>)
#  |  ├─ [io_name].out (with -wf <io_name>)
# Run with source to get the final cd to work

TEMPLATE_SRC="$HOME/main/cpp/template.cpp"
START_DIR="$PWD"

usage() {
    echo "Usage: $0 <problem_code> [-wf <io_name>]"
    echo "  <problem_code>: Problem code in CF or problem name in other OJ"
    echo "  -wf <io_name>: with explicit in and out files named <io_name>.in and <io_name>.out"
    exit 1
}

create_folder() {
    local problem_code="$1"
    
    if [[ -d "$problem_code" ]]; then
        echo "Error: Directory '$problem_code' already exists."
        return 1
    fi
    
    if ! mkdir "$problem_code"; then
        echo "Error: Failed to create directory '$problem_code'."
        return 1
    fi
    
    if ! cd "$problem_code"; then
        echo "Error: Failed to change directory to '$problem_code'."
        return 1
    fi
    
    if ! cp "$TEMPLATE_SRC" "${problem_code}.cpp"; then
        echo "Error: Failed to copy template."
        return 1
    fi
    
    return 0
}

handle_file_flag() {
    local problem_code="$1"
    local io_name="$2"
    
    if ! touch "${io_name}.in"; then
        echo "Error: Failed to create ${io_name}.in"
        return 1
    fi
    
    if ! touch "${io_name}.out"; then
        echo "Error: Failed to create ${io_name}.out"
        return 1
    fi
    
    # Uncomment everything from the template and replace
    # .in and .out with io_name.in and io_name.out
    # for old USACO submissions
    if ! sed -i 's#^\s*// ##g' "${problem_code}.cpp"; then
        echo "Error: Failed to uncomment lines in ${problem_code}.cpp"
        return 1
    fi
    
    if ! sed -i "s/\\.in/${io_name}.in/g" "${problem_code}.cpp"; then
        echo "Error: Failed to replace .in with ${io_name}.in"
        return 1
    fi
    
    if ! sed -i "s/\\.out/${io_name}.out/g" "${problem_code}.cpp"; then
        echo "Error: Failed to replace .out with ${io_name}.out"
        return 1
    fi
    
    return 0
}

handle_no_file_flag() {
    local problem_code="$1"
    
    # Delete all lines commented for OJs that require output to stdout
    # and input from stdin
    if ! sed -i '/^\s*\/\//d' "${problem_code}.cpp"; then
        echo "Error: Failed to remove commented lines from ${problem_code}.cpp"
        return 1
    fi
    
    return 0
}

# Added rest of the script to main
main() {
    if [[ ! -f "$TEMPLATE_SRC" ]]; then
        echo "Error: Template file not found at '$TEMPLATE_SRC'."
        exit 1
    fi
    
    # Check arguments existence
    if [[ -z "$1" ]]; then
        usage
    fi
    
    local problem_code="$1"
    local flag="$2"
    local io_name="$3"
    
    # Process based on flag
    case "$flag" in
        "")
            if create_folder "$problem_code"; then
                if ! handle_no_file_flag "$problem_code"; then
                    echo "Error: Failed to process template"
                    exit 1
                fi
            else
                echo "Error: Failed to create problem folder."
                exit 1
            fi
            ;;
        "-wf")
            if [[ -z "$io_name" ]]; then
                echo "Error: -wf flag requires an I/O name parameter."
                usage
            fi
            
            if create_folder "$problem_code"; then
                if ! handle_file_flag "$problem_code" "$io_name"; then
                    echo "Error: Failed to process template"
                    exit 1
                fi
            else
                echo "Error: Failed to create problem folder."
                exit 1
            fi
            ;;
        *)
            echo "Error: Invalid flag '$flag'. Use -wf <io_name> or no flag."
            usage
            ;;
    esac
    
    # Open in code and cd into created dir
    code "${START_DIR}/${problem_code}/${problem_code}.cpp" && cd "${START_DIR}/${problem_code}"
}

main "$@"