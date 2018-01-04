#!/bin/bash

# Make sure we always invoke this script from the gh-pages branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "gh-pages" ]]; then
  echo 'The doc installation script must be run from the gh-pages branch!';
  exit 1;
fi

git clean -fdx
# git clone --no-checkout --depth 1 https://github.com/tyarkoni/pliers.git docs
git checkout sphinx-docs docs
cd docs
make clean
make html
TIMESTAMP=$(date)
cp -R _build/html/* ..
cd ..
rm -rf docs
git add .
git commit -m "pliers docs built with Sphinx on $TIMESTAMP"
git push origin gh-pages