#!/usr/bin/env bash
if [[ $- != *i* ]] || [[ -z $PS1 ]]; then
  exec bash --login -i "$0" "$@"
fi

cd /home/agilex/Repo/ROSGUI-AGX

./build/bin/ROSGUIAGX