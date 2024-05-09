#!/bin/bash
username=$(whoami) # Get the current system username

for env in $(conda env list | awk '{print $1}' | grep -v '^#')
do
  if [ $env != "base" ]; then
    env_filename="env_${username}_${env}.yml"
    conda env export -n $env > $env_filename
	echo "Exported ${env} to ${env_filename}"
  fi
done
