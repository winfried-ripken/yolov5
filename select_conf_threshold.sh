#!/bin/bash

source=${source:-video.mp4}
device=${device:-cuda:0}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

echo $source $device

streamlit run app_select_conf_threshold.py -- --source $source --device $device