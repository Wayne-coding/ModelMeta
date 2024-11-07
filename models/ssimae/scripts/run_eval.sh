















if [ $
then
    echo "Usage: bash run_eval.sh [CONFIG_PATH] [DEVICE_ID]"
exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $1)
export DEVICE_ID=$2
EVAL_PATH=eval_`echo $CONFIG_PATH | rev | cut -d '/' -f 1 | rev | awk -F "_config.yaml" '{print $1}'`
if [ -d $EVAL_PATH ];
then
    rm -rf $EVAL_PATH
fi
mkdir $EVAL_PATH
cd $EVAL_PATH/ || exit
python ../eval.py --config_path=$CONFIG_PATH > eval.log 2>&1 &
tail -f eval.log
