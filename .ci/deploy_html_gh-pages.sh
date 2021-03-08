#!/bin/bash
set -euo pipefail

cd build/much-demos-such-wow/html

# disable jekyll
touch .nojekyll

git init
git add .
git commit -m "build"
git remote add origin ssh://github-brainscales2-demos_gh-pages/electronicvisions/brainscales2-demos.git
git push -f origin HEAD:gh-pages
