#!/bin/bash
set -euo pipefail

cd build/much-demos-such-wow/jupyter

# disable jekyll
touch .nojekyll

git init
git add .
git commit -m "build"
git remote add jupyter_notebooks git@github.com:electronicvisions/brainscales2-demos.git
git push -f jupyter_notebooks HEAD:jupyter_notebooks
