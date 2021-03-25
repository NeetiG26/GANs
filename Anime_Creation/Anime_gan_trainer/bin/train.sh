# Run gcloud ml-engine given a model and environment.
#
# e.g. $ app.sh encoder1 remote
# @param {model} Name of model to run (e.g. encoder1).
#                This should map to a config file named config/${model}.sh
# @param {env} Environment to run ML training (local|remote)
# @param {label} Optional label to add to the job ID


# Get params
model=$1
env=$2
label=$3

now=$(date +%Y%m%d_%H%M%S)
job_name=job_${now}_${model}_${label}

package_path=trainer/
module_name=trainer.task

config_file=config/${model}.sh
if [ ! -f "$config_file" ]; then
  echo "Config file not found: $config_file";
  exit 1;
fi

# Read config file, pass it env
# shellcheck disable=SC1090
. "$config_file"

export JOB_DIR="gs://dimensionality_reduction/models/$job_name"
# Check 'env' to run either in local or remote (CMLE)
if [ $env = 'local' ]; then
  echo 'Running job locally in the background.'
  log_file="$job_name.log"
  echo "Logging to file: $log_file"
  gcloud ml-engine local train \
    --module-name $module_name \
    --package-path $package_path \
    -- \
    $MODULE_ARGS \
    --job_dir "models/$job_name"\

elif [ $env = 'remote' ]; then
  echo 'Running job on ML engine.'
  # Create JOB_DIR environmental variable to be used in the CI nightly job
  gcloud ml-engine jobs submit training "$job_name" \
    --staging-bucket "gs://dimensionality_reduction" \
    --module-name $module_name \
    --package-path $package_path \
    --region 'us-east1'\
    --config config/config.yaml \
    --runtime-version 1.10 \
    -- \
    $MODULE_ARGS \
    --job_dir "$JOB_DIR"\

else
  echo "Unrecognized environment : $env"
  exit 1

fi
