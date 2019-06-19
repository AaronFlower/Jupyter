#!/usr/bin/env bash

for year in all/*
do
    echo -ne `basename $year .gz`"\t"
done
