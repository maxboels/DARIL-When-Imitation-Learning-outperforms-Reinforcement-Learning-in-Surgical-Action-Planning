#!/bin/bash

# Read the delete_jobs.txt file and extract job names (first column)
while read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Extract the job name (first field before whitespace)
    job_name=$(echo "$line" | awk '{print $1}')
    
    # Execute the runai delete command
    runai delete job "$job_name"
    
done < delete_jobs.txt
