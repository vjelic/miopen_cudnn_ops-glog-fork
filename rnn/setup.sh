#!/usr/bin/env bash

declare -a repos=(
    "git@github.com:johnpzh/cudnn_samples_v8.git"
)

# echo "${repos[@]}"
for repo in "${repos[@]}"; do
    repo_name="$(basename ${repo} | sed -E 's/\.git//g')";

    if [ ! -d "${repo_name}" ]; then
        git clone --recursive ${repo}
        echo
    else
        echo "${repo_name} found and skipped..."
        echo
    fi
done

sed -i '41s/.*/SMS ?= 90/' cudnn_samples_v8/samples_common.mk
