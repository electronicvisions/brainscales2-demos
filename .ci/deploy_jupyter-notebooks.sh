#!/bin/bash
set -euo pipefail

cd build/much-demos-such-wow/jupyter

# disable jekyll
touch .nojekyll

git init
git add .
git commit -m "build"
git remote add origin git@github.com:electronicvisions/brainscales2-demos.git
git push -f origin HEAD:jupyter-notebooks
