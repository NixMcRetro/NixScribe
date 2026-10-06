#!/bin/zsh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
./build_macos_app.sh
printf '\nPress Return to close this window… '
read _
