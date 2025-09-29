#!/bin/bash

ruff check --fix

mypy --install-types --non-interactive src/