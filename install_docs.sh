#!/bin/bash

# git clone --no-checkout --depth 1 https://github.com/tyarkoni/pliers.git docs
git checkout sphinx-docs docs
ls
cd docs
make clean
TIMESTAMP=$(date)
make html
ls _build/html/*
cp -R _build/html/* ..
cd ..
ls
rm -rf docs
git add .
git commit -m "pliers docs built with Sphinx on $TIMESTAMP"
git push origin gh-pages