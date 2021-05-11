#!/bin/bash
docker run --runtime nvidia --rm -it -v `pwd`:/work -w /work tsa python3 "$@"