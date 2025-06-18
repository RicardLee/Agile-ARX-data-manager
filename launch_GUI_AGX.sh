#!/usr/bin/env bash

if [[ $- != *i* ]] || [[ -z $PS1 ]]; then
  exec bash --login -i "$0" "$@"
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

cd "$SCRIPT_DIR" 

./ui/ROSGUI-AGX/build/bin/ROSGUIAGX
